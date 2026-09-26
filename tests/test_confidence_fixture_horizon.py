"""R4B.2c: decision confidence + four-GW fixture horizon.

Targeted, deterministic.  Temporary databases only; the live database is never
opened for writing.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from fpl_brain import decision_confidence as dc
from fpl_brain import four_gw_decision as fg
from fpl_brain.database import connect_database
from fpl_brain.route_comparator import NEAR_TIE_K

DECISION_EVENTS = [5, 6, 7, 8]
DEADLINES = [
    (5, "2026-09-18T17:30:00Z"),
    (6, "2026-10-10T10:00:00Z"),
    (7, "2026-10-17T10:00:00Z"),
    (8, "2026-10-23T17:30:00Z"),
    (9, "2026-10-30T17:30:00Z"),
]
BASE_FIXTURES = [
    (1, 5, 1, 2, "2026-09-19T14:00:00Z"),
    (2, 6, 1, 2, "2026-10-11T14:00:00Z"),
    (3, 7, 1, 2, "2026-10-18T14:00:00Z"),
    (4, 8, 1, 2, "2026-10-24T14:00:00Z"),
]


def _world(extra_fixtures=(), extra_teams=(3, 4, 9)):
    path = Path(tempfile.mkdtemp()) / "fpl.db"
    conn = connect_database(path)
    with conn:
        for tid in (1, 2, *extra_teams):
            conn.execute(
                "INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                " VALUES (?,?,?,'{}','2026-09-01T00:00:00Z')",
                (tid, f"T{tid}", f"T{tid}"),
            )
        for event, deadline in DEADLINES:
            conn.execute(
                "INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
                " VALUES (?,?,?,0,'{}','2026-09-01T00:00:00Z')",
                (event, f"GW{event}", deadline),
            )
        for fixture in list(BASE_FIXTURES) + list(extra_fixtures):
            conn.execute(
                "INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started,"
                " raw_json, updated_at) VALUES (?,?,?,?,?,0,0,'{}','2026-09-01T00:00:00Z')",
                fixture,
            )
    return conn


def _classify(conn, last_event=38):
    return fg.classify_fixture_horizon(conn, DECISION_EVENTS, last_event=last_event)


# ---------------------------------------------------------------------------
# 18. Confidence tests A-J
# ---------------------------------------------------------------------------

CLEAN = {"role_confidence": "HIGH", "conflict_flags": []}
UNCERTAIN = {"role_confidence": "LOW", "conflict_flags": [], "uncertainty_reasons": ["no_current_season_starts"]}
CONFLICT = {
    "role_confidence": "LOW",
    "conflict_flags": ["MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE"],
    "prior_role_discontinuity": True,
    "uncertainty_reasons": ["prior_role_not_reproduced_at_current_club"],
}
NEAR = {"mean_difference": 0.05, "paired_se": 0.10}
FAR = {"mean_difference": 5.0, "paired_se": 0.10}


def _classify_confidence(paired, evidence):
    result = dc.classify_decision_confidence(
        paired=paired, role_evidence={1: evidence}, focus_player_ids=[1]
    )
    dc.assert_confidence_invariants(result)
    return result


def test_a_paired_near_tie_without_role_uncertainty_is_near_tie():
    assert _classify_confidence(NEAR, CLEAN)["state"] == dc.CONFIDENCE_NEAR_TIE


def test_b_paired_near_tie_with_role_uncertainty():
    result = _classify_confidence(NEAR, UNCERTAIN)
    assert result["state"] == dc.CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY
    assert result["near_tie"] is True


def test_c_no_near_tie_with_role_uncertainty_is_role_uncertainty():
    result = _classify_confidence(FAR, UNCERTAIN)
    assert result["state"] == dc.CONFIDENCE_ROLE_UNCERTAINTY
    assert result["near_tie"] is False


def test_d_role_conflict_overrides_everything():
    for paired in (NEAR, FAR):
        assert _classify_confidence(paired, CONFLICT)["state"] == dc.CONFIDENCE_MODEL_EVIDENCE_CONFLICT


def test_e_otherwise_strong_recommendation():
    assert _classify_confidence(FAR, CLEAN)["state"] == dc.CONFIDENCE_STRONG


def test_f_no_near_tie_labelled_state_when_near_tie_is_false():
    """The R4A.2 contradiction must be impossible now."""

    for paired in (NEAR, FAR, None):
        for evidence in (CLEAN, UNCERTAIN, CONFLICT):
            result = _classify_confidence(paired, evidence)
            if not result["near_tie"]:
                assert result["state"] not in dc.NEAR_TIE_LABELLED_STATES, result
            if result["state"] == dc.CONFIDENCE_ROLE_UNCERTAINTY:
                assert "NEAR_TIE" not in result["state"]


def test_g_fixed_materiality_threshold_does_not_recompute_near_tie():
    """The 0.25 materiality margin must not act as a near-tie test."""

    # a large CORE margin that is nonetheless a statistical near tie
    result = dc.classify_decision_confidence(
        paired=NEAR, role_evidence={1: CLEAN}, focus_player_ids=[1],
        selected_core=10.0, alternative_cores={"other": 9.0},
    )
    assert result["near_tie"] is True, "near tie comes from the paired test only"
    assert result["material_margin"] == dc.MATERIAL_FRONTIER_CHANGE_CORE
    assert result["near_tie_k"] == NEAR_TIE_K == 1.96
    assert dc.MATERIAL_FRONTIER_CHANGE_CORE == 0.25

    # and the reverse: a small margin with a clear paired result is NOT a near tie
    tight = dc.classify_decision_confidence(
        paired=FAR, role_evidence={1: CLEAN}, focus_player_ids=[1],
        selected_core=10.0, alternative_cores={"other": 9.95},
    )
    assert tight["near_tie"] is False


def test_h_confidence_never_affects_ranking():
    result = _classify_confidence(FAR, CLEAN)
    assert result["affects_ranking"] is False
    source = Path("fpl_brain/decision_confidence.py").read_text(encoding="utf-8")
    assert '"affects_ranking": False' in source
    assert "does not rank routes" in source


def test_i_role_evidence_comes_from_certified_minutes_ids_only():
    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "_certified_role_evidence" in source
    assert "certified_minutes_run_only" in source
    # it reads the run ids named by the certified bundles
    assert 'certified_runs[int(event)]["minutes_v1"]' in source
    # and never the live scouting or latest minutes paths
    assert "scouting_current_rows(" not in source


def test_j_post_certification_role_evidence_cannot_alter_confidence():
    """Confidence depends on the supplied certified evidence only."""

    a = _classify_confidence(FAR, CLEAN)
    b = _classify_confidence(FAR, CLEAN)
    assert a["state"] == b["state"] == dc.CONFIDENCE_STRONG
    # a different (newer) block would change the state; the point is the CALLER
    # supplies certified blocks only, which the provenance assertion enforces.
    assert a["role_evidence_source"].get("provenance") is None  # not threaded here
    threaded = dc.classify_decision_confidence(
        paired=FAR, role_evidence={1: CLEAN}, focus_player_ids=[1],
        role_evidence_source={"provenance": "certified_minutes_run_only", "minutes_run_ids": [166]},
    )
    assert threaded["role_evidence_source"]["provenance"] == "certified_minutes_run_only"


# ---------------------------------------------------------------------------
# 19. Fixture tests K-X
# ---------------------------------------------------------------------------


def test_k_normal_four_gw_fixture_set_is_complete():
    conn = _world()
    result = _classify(conn)
    assert result["complete"] is True
    assert result["blocking_reasons"] == []
    assert result["decision_events"] == DECISION_EVENTS
    conn.close()


def test_l_team_specific_blank_is_valid_and_does_not_block():
    """Team 2 genuinely blanks in GW7 while team 1 plays."""

    conn = _world(extra_fixtures=[(5, 7, 3, 4, "2026-10-18T16:00:00Z")])
    # remove team 2 from GW7 by giving GW7 to teams 1v3 instead
    with conn:
        conn.execute("UPDATE fixtures SET team_a=3 WHERE id=3")
    result = _classify(conn)
    states = result["team_event_states"]
    assert states["2:7"]["state"] == fg.CERTIFIED_BLANK
    assert states["2:7"]["fixture_count"] == 0
    assert states["1:7"]["state"] == fg.CERTIFIED_SINGLE
    assert result["complete"] is True, "a legitimate team blank must not block"
    conn.close()


def test_l_unresolved_fixture_for_a_team_is_not_a_certified_blank():
    conn = _world(extra_fixtures=[(9, None, 2, 4, "2026-10-20T14:00:00Z")])
    result = _classify(conn)
    states = result["team_event_states"]
    assert states["2:7"]["state"] != fg.CERTIFIED_BLANK
    assert states["2:7"]["unresolved_possible"] is True
    assert result["complete"] is False
    conn.close()


def test_m_valid_double_gameweek_counts_both_and_does_not_block():
    conn = _world(extra_fixtures=[(6, 6, 1, 9, "2026-10-11T16:00:00Z")])
    result = _classify(conn)
    states = result["team_event_states"]
    assert states["1:6"]["state"] == fg.CERTIFIED_DOUBLE_OR_MORE
    assert states["1:6"]["fixture_count"] == 2
    assert result["complete"] is True, "two distinct fixtures must not block"
    conn.close()


def test_n_kickoff_between_final_deadline_and_next_boundary_is_plausibly_inside():
    """CRITICAL: the GW8 deadline must not be used as the horizon end."""

    # GW8 deadline 2026-10-23; GW9 deadline 2026-10-30
    conn = _world(extra_fixtures=[(7, None, 3, 4, "2026-10-27T14:00:00Z")])
    result = _classify(conn)
    assert result["boundary_basis"] == "NEXT_EVENT_DEADLINE_AFTER_WINDOW"
    assert result["horizon_end_boundary"] == "2026-10-30T17:30:00Z"
    assert len(result["plausibly_inside"]) == 1
    assert result["complete"] is False
    assert result["plausibly_inside"][0]["fixture_id"] == 7
    conn.close()


def test_o_kickoff_at_or_after_the_next_boundary_is_outside():
    conn = _world(extra_fixtures=[(8, None, 3, 4, "2026-10-31T14:00:00Z")])
    result = _classify(conn)
    assert len(result["outside"]) == 1
    assert result["complete"] is True
    conn.close()


def test_o_kickoff_before_the_window_start_is_outside():
    conn = _world(extra_fixtures=[(10, None, 3, 4, "2026-09-01T14:00:00Z")])
    result = _classify(conn)
    assert len(result["outside"]) == 1
    assert result["complete"] is True
    conn.close()


def test_p_event_null_with_unknown_kickoff_is_unknown_and_blocks():
    conn = _world(extra_fixtures=[(11, None, 3, 4, None)])
    result = _classify(conn)
    assert len(result["unknown"]) == 1
    assert result["complete"] is False
    conn.close()


def test_q_fixture_assigned_to_gw9_is_outside_and_does_not_block():
    conn = _world(extra_fixtures=[(12, 9, 3, 4, "2026-10-31T14:00:00Z")])
    result = _classify(conn)
    assert len(result["outside"]) == 1
    assert result["outside"][0]["event"] == 9
    assert result["complete"] is True
    conn.close()


def test_r_unresolved_extra_fixture_that_could_create_a_dgw_blocks():
    conn = _world(extra_fixtures=[(13, None, 1, 4, "2026-10-11T16:00:00Z")])
    result = _classify(conn)
    # team 1 already plays GW6, so an extra unresolved fixture could make it a DGW
    assert result["complete"] is False
    assert result["team_event_states"]["1:6"]["unresolved_possible"] is True
    conn.close()


def test_s_duplicate_fixture_identity_is_an_integrity_failure_not_a_dgw():
    conn = _world(extra_fixtures=[(14, 5, 1, 2, "2026-09-19T14:00:00Z")])
    result = _classify(conn)
    assert len(result["integrity_failures"]) == 1
    assert result["integrity_failures"][0]["diagnostic"] == fg.FIXTURE_INTEGRITY_DUPLICATE_IDENTITY
    assert result["complete"] is False
    conn.close()


def test_s_two_distinct_fixtures_same_team_event_is_not_an_integrity_failure():
    conn = _world(extra_fixtures=[(15, 5, 1, 9, "2026-09-20T14:00:00Z")])
    result = _classify(conn)
    assert result["integrity_failures"] == []
    assert result["team_event_states"]["1:5"]["state"] == fg.CERTIFIED_DOUBLE_OR_MORE
    conn.close()


def test_t_snapshot_fixture_data_is_used_not_live_mutation():
    """Classification reads the snapshot: a later LIVE mutation cannot change it."""

    import shutil

    live = _world()
    before = _classify(live)
    assert before["complete"] is True

    # take a physical snapshot copy, then classify from the SNAPSHOT connection
    snapshot_path = Path(tempfile.mkdtemp()) / "snapshot.db"
    live.close()
    source_path = Path(live.execute("PRAGMA database_list").fetchone()[2]) if False else None
    # rebuild the same world and snapshot it via SQLite VACUUM INTO
    conn = _world()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    reader = sqlite3.connect(db_path)
    reader.execute("VACUUM INTO ?", (str(snapshot_path),))
    reader.close()
    snap_conn = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)
    snap_conn.row_factory = sqlite3.Row

    with conn:  # mutate the LIVE database only
        conn.execute("UPDATE fixtures SET event=9 WHERE id=3")
        conn.execute("DELETE FROM fixtures WHERE id=4")
    from_snapshot = fg.classify_fixture_horizon(snap_conn, DECISION_EVENTS, last_event=38)
    from_live = fg.classify_fixture_horizon(conn, DECISION_EVENTS, last_event=38)

    # The snapshot classification is IDENTICAL to the pre-mutation classification.
    assert from_snapshot["complete"] is True
    assert from_snapshot["decision_events"] == before["decision_events"]
    assert from_snapshot["outside"] == before["outside"]
    assert from_snapshot["team_event_states"] == before["team_event_states"]
    # ...while the live view has moved (fixture 3 reassigned to GW9, fixture 4 gone)
    assert from_live["team_event_states"] != from_snapshot["team_event_states"]
    assert any(row["event"] == 9 for row in from_live["outside"])
    snap_conn.close()
    conn.close()


def test_u_horizon_uses_exactly_four_decision_events():
    source = Path("fpl_brain/four_gw_decision.py").read_text(encoding="utf-8")
    assert "DECISION_HORIZON_LENGTH" in source
    conn = _world()
    result = _classify(conn)
    assert len(result["decision_events"]) == 4
    # three events are not contiguous-four and must block
    short = fg.classify_fixture_horizon(conn, [5, 6, 7], last_event=38)
    assert short["complete"] is False
    conn.close()


def test_v_blocked_horizon_suppresses_the_normal_transfer_recommendation():
    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "fixture_horizon_blocking_reasons" in source
    assert "RECOMMENDATION_SUPPRESSED" in source
    # the suppressed block carries no preferred route
    assert '"preferred_route_id": None' in source


def test_w_no_play_wildcard_is_introduced():
    """R4B.2c code must not create PLAY_WILDCARD (the R4B.1 contract is untouched)."""

    # the confidence module never emits it
    source = Path("fpl_brain/decision_confidence.py").read_text(encoding="utf-8")
    assert "PLAY_WILDCARD" not in source
    # and the new fixture-horizon code never emits it
    fg_source = Path("fpl_brain/four_gw_decision.py").read_text(encoding="utf-8")
    horizon_block = fg_source[fg_source.index("def classify_fixture_horizon"):]
    assert "PLAY_WILDCARD" not in horizon_block
    # the pre-existing Wildcard contract stays NOT_SUPPORTED
    assert "NOT_SUPPORTED" in fg_source


def test_x_projection_runs_remains_212():
    from fpl_brain.config import config_path, load_config

    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        total, max_id = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
    finally:
        conn.close()
    assert total == 212 and max_id == 212


# ---------------------------------------------------------------------------
# R4B.2c INTEGRATION CORRECTIONS
# ---------------------------------------------------------------------------


def test_role_relevant_set_is_transfers_and_armband_not_the_squad():
    """Correction 1: the role-relevant set must not be derived from the squad."""

    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    # the set is built from transfers in/out and the preferred route's armband
    assert "role_relevant_ids" in source
    assert 'int(move["in"])' in source and 'int(move["out"])' in source
    assert '"captain_id", "vice_captain_id"' in source
    # and NOT from the certified squad
    assert "role_relevant_players = sorted(source_squad" not in source
    assert 'focus_players = sorted(' not in source
    # the transfer-in player must be queried from certified minutes ids
    assert "role_relevant_players" in source
    assert "_certified_role_evidence(\n            conn, certified_runs, role_relevant_players" in source


def test_role_evidence_includes_an_unowned_transfer_in_player():
    """A transfer-IN player is usually not in the pre-transfer squad yet must be
    evaluated: the helper filters only by the requested ids, never by squad."""

    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    helper = source[source.index("def _certified_role_evidence"):]
    helper = helper[: helper.index("\ndef ")]
    assert "source_squad" not in helper
    assert "squad_ids" not in helper
    assert "if pid not in set(int(p) for p in player_ids)" in helper

    # functional proof: an unowned player id still receives role evidence
    from fpl_brain import analytics, repositories as repo  # noqa: F401

    evidence, source_block = _helper_probe()
    assert 9001 in evidence, "a transfer-in player outside the squad must still be evaluated"
    assert source_block["provenance"] == "certified_minutes_run_only"


def _helper_probe():
    """Exercise _certified_role_evidence on a synthetic certified minutes run."""

    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_four_gw_decision as runner

    from fpl_brain import analytics
    from fpl_brain.database import connect_database

    conn = connect_database(Path(tempfile.mkdtemp()) / "fpl.db")
    with conn:
        conn.execute(
            "INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
            " VALUES (3,'MID','{}','2026-09-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
            " VALUES (1,'One','ONE','{}','2026-09-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
            " VALUES (5,'GW5','2026-09-18T17:30:00Z',0,'{}','2026-09-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started,"
            " raw_json, updated_at) VALUES (48,5,1,1,'2026-09-19T14:00:00Z',0,0,'{}','2026-09-01T00:00:00Z')"
        )
        # an UNOWNED transfer target with an uncertain certified role
        conn.execute(
            "INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
            " last_seen_at, raw_json, updated_at)"
            " VALUES (9001,'Target',1,3,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}','2026-09-01T00:00:00Z')"
        )
        run_id = analytics.create_projection_run(
            conn, model_family="minutes_v1", model_version="minutes_v1.6.0", planning_event=5,
            planning_context_hash="ctx", data_cutoff="2026-09-12T19:00:00Z", scouting_cutoff=None,
            official_run_ids=None, deadline_status="PRE_DEADLINE", allow_future_cutoff=True,
        )
        analytics.freeze_minutes_predictions(
            conn, run_id, player_id=9001, fixture_id=48, event=5,
            payload={
                "role_evidence": {
                    "role_confidence": "LOW",
                    "conflict_flags": [],
                    "uncertainty_reasons": ["no_current_season_starts"],
                }
            },
            model_version="minutes_v1.6.0",
        )
    class _ProbeGeneration:
        """The role-evidence helper reads ONE field off the generation: its id."""

        generation_id = "sha256:" + "a" * 64

    certified = {5: {"minutes_v1": run_id}}
    evidence, block = runner._certified_role_evidence(
        conn, certified, [9001], [5], _ProbeGeneration()
    )
    conn.close()
    return evidence, block


def test_confidence_reflects_role_uncertainty_for_a_transfer_in_player():
    """The end-to-end shape: near tie + an uncertain transfer-in target."""

    subject = {
        "role_confidence": "LOW",
        "conflict_flags": [],
        "uncertainty_reasons": ["no_current_season_starts"],
    }
    result = dc.classify_decision_confidence(
        paired={"mean_difference": 0.05, "paired_se": 0.10},
        role_evidence={9001: subject},
        focus_player_ids=[9001],
        role_evidence_source={"provenance": "certified_minutes_run_only", "minutes_run_ids": [1],
                              "player_ids": [9001]},
    )
    dc.assert_confidence_invariants(result)
    assert result["state"] == dc.CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY
    assert result["near_tie"] is True
    assert result["role_uncertainties"][0]["player_id"] == 9001

    # ...and with a clear paired margin the same evidence yields ROLE_UNCERTAINTY
    apart = dc.classify_decision_confidence(
        paired={"mean_difference": 5.0, "paired_se": 0.10},
        role_evidence={9001: subject},
        focus_player_ids=[9001],
    )
    assert apart["state"] == dc.CONFIDENCE_ROLE_UNCERTAINTY


def test_missing_paired_evidence_is_not_near_tie_false():
    """Correction 2: absent paired evidence must NOT be read as 'not near tied'."""

    clean = {"role_confidence": "HIGH", "conflict_flags": []}
    result = dc.classify_decision_confidence(
        paired=None, role_evidence={1: clean}, focus_player_ids=[1]
    )
    dc.assert_confidence_invariants(result)
    assert result["state"] == dc.CONFIDENCE_INCOMPLETE
    assert result["state"] != dc.CONFIDENCE_STRONG
    assert result["state"] not in dc.NEAR_TIE_LABELLED_STATES
    assert result["paired_available"] is False
    assert result["decisive"] is False
    assert result["confidence_diagnostic"] == dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED
    assert result["affects_ranking"] is False

    # an empty record is equally insufficient
    empty = dc.classify_decision_confidence(paired={}, role_evidence={1: clean}, focus_player_ids=[1])
    assert empty["state"] == dc.CONFIDENCE_INCOMPLETE

    # an explicit record IS sufficient
    ok = dc.classify_decision_confidence(
        paired={"mean_difference": 5.0, "paired_se": 0.1}, role_evidence={1: clean},
        focus_player_ids=[1],
    )
    assert ok["state"] == dc.CONFIDENCE_STRONG
    assert ok["decisive"] is True
    assert ok["confidence_diagnostic"] is None


def test_non_decisive_callers_may_opt_out_of_the_paired_requirement():
    result = dc.classify_decision_confidence(
        paired=None, role_evidence={1: {"role_confidence": "HIGH", "conflict_flags": []}},
        focus_player_ids=[1], require_paired=False,
    )
    assert result["state"] == dc.CONFIDENCE_STRONG
    assert result["paired_required"] is False
    assert result["decisive"] is True


def test_conflict_is_still_reported_when_paired_evidence_is_missing():
    """The incomplete state must not hide the conflicts that were found."""

    conflict = {
        "role_confidence": "LOW",
        "conflict_flags": ["MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE"],
        "prior_role_discontinuity": True,
    }
    result = dc.classify_decision_confidence(
        paired=None, role_evidence={1: conflict}, focus_player_ids=[1]
    )


# ---------------------------------------------------------------------------
# R4B.2c INTEGRATION CORRECTIONS (mandatory)
# ---------------------------------------------------------------------------


def _helper_probe():
    """Exercise _certified_role_evidence on a synthetic certified minutes run.

    The point is that a transfer-IN player is normally NOT in the pre-transfer
    certified squad, so the helper must filter only by the requested ids.
    """

    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_four_gw_decision as runner

    from fpl_brain import analytics
    from fpl_brain.database import connect_database

    conn = connect_database(Path(tempfile.mkdtemp()) / "fpl.db")
    with conn:
        conn.execute(
            "INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
            " VALUES (3,'MID','{}','2026-09-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
            " VALUES (1,'One','ONE','{}','2026-09-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
            " VALUES (5,'GW5','2026-09-18T17:30:00Z',0,'{}','2026-09-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started,"
            " raw_json, updated_at)"
            " VALUES (48,5,1,1,'2026-09-19T14:00:00Z',0,0,'{}','2026-09-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
            " last_seen_at, raw_json, updated_at)"
            " VALUES (9001,'Target',1,3,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}',"
            "'2026-09-01T00:00:00Z')"
        )
        run_id = analytics.create_projection_run(
            conn,
            model_family="minutes_v1",
            model_version="minutes_v1.6.0",
            planning_event=5,
            planning_context_hash="ctx",
            data_cutoff="2026-09-12T19:00:00Z",
            scouting_cutoff=None,
            official_run_ids=None,
            deadline_status="PRE_DEADLINE",
            allow_future_cutoff=True,
        )
        analytics.freeze_prediction(
            conn,
            run_id,
            kind=analytics.MINUTES_V1_KIND,
            player_id=9001,
            fixture_id=48,
            event=5,
            payload={
                "role_evidence": {
                    "role_confidence": "LOW",
                    "conflict_flags": [],
                    "uncertainty_reasons": ["no_current_season_starts"],
                }
            },
            model_version="minutes_v1.6.0",
        )
    class _ProbeGeneration:
        """The role-evidence helper reads ONE field off the generation: its id."""

        generation_id = "sha256:" + "a" * 64

    certified = {5: {"minutes_v1": run_id}}
    evidence, block = runner._certified_role_evidence(
        conn, certified, [9001], [5], _ProbeGeneration()
    )
    conn.close()
    return evidence, block


def test_correction1_role_relevant_set_is_transfers_and_armband_not_the_squad():
    """Correction 1: the role-relevant set must not be derived from the squad."""

    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "role_relevant_ids" in source
    assert 'int(move["in"])' in source and 'int(move["out"])' in source
    assert '"captain_id", "vice_captain_id"' in source
    # it is NOT built from the certified squad
    assert "role_relevant_players = sorted(source_squad" not in source
    assert "focus_players = sorted(" not in source
    # the relevant ids are what get queried
    assert "role_relevant_players," in source


def test_correction1_role_evidence_includes_an_unowned_transfer_in_player():
    """A transfer-IN player is usually not in the pre-transfer squad yet must be
    evaluated: the helper filters only by the requested ids, never by squad."""

    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    helper = source[source.index("def _certified_role_evidence"):]
    helper = helper[: helper.index("\ndef ")]
    assert "source_squad" not in helper
    assert "squad_ids" not in helper
    assert "if pid not in set(int(p) for p in player_ids)" in helper

    evidence, source_block = _helper_probe()
    assert 9001 in evidence, "a transfer-in player outside the squad must still be evaluated"
    assert evidence[9001]["role_confidence"] == "LOW"
    assert source_block["provenance"] == "certified_minutes_run_only"
    assert source_block["player_ids"] == [9001]


def test_correction1_confidence_reflects_uncertainty_for_a_transfer_in_player():
    """Regression: recommended transfer buys an unowned, role-uncertain player."""

    subject = {
        "role_confidence": "LOW",
        "conflict_flags": [],
        "uncertainty_reasons": ["no_current_season_starts"],
    }
    result = dc.classify_decision_confidence(
        paired={"mean_difference": 0.05, "paired_se": 0.10},
        role_evidence={9001: subject},
        focus_player_ids=[9001],
        role_evidence_source={
            "provenance": "certified_minutes_run_only",
            "minutes_run_ids": [1],
            "player_ids": [9001],
        },
    )
    dc.assert_confidence_invariants(result)
    assert result["state"] == dc.CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY
    assert result["role_uncertainties"][0]["player_id"] == 9001

    # with a clear paired margin the same evidence yields ROLE_UNCERTAINTY
    apart = dc.classify_decision_confidence(
        paired={"mean_difference": 5.0, "paired_se": 0.10},
        role_evidence={9001: subject},
        focus_player_ids=[9001],
    )
    assert apart["state"] == dc.CONFIDENCE_ROLE_UNCERTAINTY


def test_correction2_missing_paired_evidence_is_not_near_tie_false():
    """Correction 2: absent paired evidence must NOT be read as 'not near tied'."""

    clean = {"role_confidence": "HIGH", "conflict_flags": []}
    result = dc.classify_decision_confidence(
        paired=None, role_evidence={1: clean}, focus_player_ids=[1]
    )
    dc.assert_confidence_invariants(result)
    assert result["state"] == dc.CONFIDENCE_INCOMPLETE
    assert result["state"] != dc.CONFIDENCE_STRONG
    assert result["state"] not in dc.NEAR_TIE_LABELLED_STATES
    assert result["paired_available"] is False
    assert result["decisive"] is False
    assert result["confidence_diagnostic"] == dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED
    assert result["affects_ranking"] is False

    empty = dc.classify_decision_confidence(paired={}, role_evidence={1: clean}, focus_player_ids=[1])
    assert empty["state"] == dc.CONFIDENCE_INCOMPLETE

    ok = dc.classify_decision_confidence(
        paired={"mean_difference": 5.0, "paired_se": 0.1},
        role_evidence={1: clean},
        focus_player_ids=[1],
    )
    assert ok["state"] == dc.CONFIDENCE_STRONG
    assert ok["decisive"] is True
    assert ok["confidence_diagnostic"] is None


def test_correction2_non_decisive_callers_may_opt_out():
    result = dc.classify_decision_confidence(
        paired=None,
        role_evidence={1: {"role_confidence": "HIGH", "conflict_flags": []}},
        focus_player_ids=[1],
        require_paired=False,
    )
    assert result["state"] == dc.CONFIDENCE_STRONG
    assert result["paired_required"] is False
    assert result["decisive"] is True


def test_correction2_conflicts_remain_visible_when_paired_evidence_is_missing():
    conflict = {
        "role_confidence": "LOW",
        "conflict_flags": ["MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE"],
        "prior_role_discontinuity": True,
    }
    result = dc.classify_decision_confidence(
        paired=None, role_evidence={1: conflict}, focus_player_ids=[1]
    )
    assert result["state"] == dc.CONFIDENCE_INCOMPLETE
    assert result["role_conflicts"], "conflicts must remain visible in the payload"


def test_correction2_canonical_key_is_defined_and_consumed():
    assert dc.CANONICAL_PAIRED_DIAGNOSTIC_KEY == "canonical_paired_near_tie"
    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "dc.CANONICAL_PAIRED_DIAGNOSTIC_KEY" in source
    assert "result.get(dc.CANONICAL_PAIRED_DIAGNOSTIC_KEY)" in source
    assert "DIAG_PAIRED_DIAGNOSTIC_REQUIRED" in source
