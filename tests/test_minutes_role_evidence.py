"""R2B: minutes evidence taxonomy, role-change correctness, and counter-tests.

Deterministic. Builds synthetic worlds in temporary databases and asserts
*relational* properties rather than arbitrary exact probabilities, except where
an exact value follows analytically from the model's own arithmetic.
"""

from __future__ import annotations

import pytest

from fpl_brain import minutes_model
from fpl_brain.database import connect_database
from fpl_brain.models import PlayerGameweekRecord, PlayerSeasonHistoryRecord

from test_minutes_model import CUTOFF, analytics_repo_upserts

GK_TEAM = 1
FIXTURE_EVENT_4_TEAM_1 = 4


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------


def _gk_world(conn, subjects: dict[int, dict]):
    """Seed a one-team world of goalkeepers.

    ``subjects`` maps player_id -> {
        status, pgw: [(event, fixture, minutes, starts)], prior: (starts, minutes) | None,
        chance, scout: (key, value) | None, snapshots: [(event_context, status)] | None
    }
    """
    players = {pid: (GK_TEAM, 1, spec.get("status", "a")) for pid, spec in subjects.items()}
    pgw: dict[int, list] = {}
    snapshots: dict[int, tuple] = {}
    scout: dict[int, tuple] = {}
    for pid, spec in subjects.items():
        pgw[pid] = spec.get("pgw", [])
        snapshots[pid] = (spec.get("status", "a"), spec.get("chance"))
        if spec.get("scout"):
            scout[pid] = spec["scout"]
    seed = {"players": players, "pgw": pgw, "snapshots": snapshots, "scout_notes": scout}
    with conn:
        analytics_repo_upserts(conn, seed)
        history_rows = []
        for pid, spec in subjects.items():
            prior = spec.get("prior")
            if prior:
                history_rows.append(
                    PlayerSeasonHistoryRecord(
                        player_id=pid,
                        season_name="2025/26",
                        minutes=int(prior[1]),
                        starts=int(prior[0]),
                        total_points=100,
                        raw_json={},
                    )
                )
        if history_rows:
            from fpl_brain import repositories as repo

            repo.upsert_player_season_histories(conn, history_rows, observed_at="2026-09-10T08:00:00Z")
    return seed


def _rows(conn, config=None):
    config = config or minutes_model.MinutesModelConfig()
    return {
        row["player_id"]: row
        for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)
        if row["fixture_id"] == FIXTURE_EVENT_4_TEAM_1
    }


def _unused(event: int, fixture: int) -> tuple:
    return (event, fixture, 0, 0)


# The Sep-12 shape: a prior-club regular who is an unused substitute three times
# while a competitor starts all three.
TRANSFER_BACKUP = {
    60: {
        "status": "a",
        "pgw": [_unused(1, 1), _unused(2, 2), _unused(3, 3)],
        "prior": (35, 3150),
        "scout": ("role_security_5gw", "very_low"),
    },
    61: {
        "status": "a",
        "pgw": [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 90, 1)],
    },
}


# ---------------------------------------------------------------------------
# Evidence taxonomy — pure classification
# ---------------------------------------------------------------------------


def test_started_and_bench_and_unused_and_unavailable_are_distinguished():
    rows = [
        {"fixture_event": 1, "minutes": 90, "starts": 1},
        {"fixture_event": 2, "minutes": 20, "starts": 0},
        {"fixture_event": 3, "minutes": 0, "starts": 0},
        {"fixture_event": 4, "minutes": None, "starts": None},
    ]
    history = [
        {"id": 1, "event_context": 1, "captured_at": "2026-08-20T00:00:00Z", "status": "a"},
        {"id": 2, "event_context": 2, "captured_at": "2026-08-27T00:00:00Z", "status": "a"},
        {"id": 3, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": "a"},
        {"id": 4, "event_context": 4, "captured_at": "2026-09-10T00:00:00Z", "status": "i"},
    ]
    out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
    assert [row["evidence_class"] for row in out] == [
        minutes_model.EVIDENCE_STARTED,
        minutes_model.EVIDENCE_BENCH_APPEARANCE,
        minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES,
        minutes_model.EVIDENCE_UNKNOWN,
    ]


def test_zero_minutes_with_unavailable_status_is_not_role_evidence():
    rows = [{"fixture_event": 3, "minutes": 0, "starts": 0}]
    for status in ("i", "u", "s", "n", "d"):
        history = [{"id": 1, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": status}]
        out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
        assert out[0]["evidence_class"] == minutes_model.EVIDENCE_UNAVAILABLE, status


def test_zero_minutes_with_zero_chance_is_unavailable():
    rows = [{"fixture_event": 3, "minutes": 0, "starts": 0}]
    history = [
        {"id": 1, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": "a",
         "chance_of_playing_this_round": 0}
    ]
    out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
    assert out[0]["evidence_class"] == minutes_model.EVIDENCE_UNAVAILABLE


def test_no_availability_trail_never_becomes_negative_evidence():
    """UNKNOWN must not silently become negative evidence."""

    rows = [{"fixture_event": 3, "minutes": 0, "starts": 0}]
    out = minutes_model.classify_evidence_rows(rows, snapshot_history=None)
    assert out[0]["evidence_class"] == minutes_model.EVIDENCE_UNKNOWN
    assert "no attributable availability evidence" in out[0]["evidence_reason"]


def test_varying_trail_without_event_context_is_unknown_not_guessed():
    rows = [{"fixture_event": 3, "minutes": 0, "starts": 0}]
    history = [
        {"id": 1, "event_context": None, "captured_at": "2026-08-20T00:00:00Z", "status": "a"},
        {"id": 2, "event_context": None, "captured_at": "2026-09-05T00:00:00Z", "status": "i"},
    ]
    out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
    assert out[0]["evidence_class"] == minutes_model.EVIDENCE_UNKNOWN


def test_constant_trail_without_event_context_is_attributed():
    rows = [{"fixture_event": 3, "minutes": 0, "starts": 0}]
    history = [
        {"id": 1, "event_context": None, "captured_at": "2026-08-20T00:00:00Z", "status": "a"},
        {"id": 2, "event_context": None, "captured_at": "2026-09-05T00:00:00Z", "status": "a"},
    ]
    out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
    assert out[0]["evidence_class"] == minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES
    assert out[0]["availability_basis"] == "assumed_constant_status"


def test_matchday_squad_class_is_reserved_and_never_inferred():
    """The official payload has no matchday-squad field, so the class is unreachable."""

    rows = [{"fixture_event": 3, "minutes": 0, "starts": 0}]
    history = [{"id": 1, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": "a"}]
    out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
    assert out[0]["evidence_class"] != minutes_model.EVIDENCE_NOT_IN_MATCHDAY_SQUAD


# ---------------------------------------------------------------------------
# Dubravka regression (the Sep-12 shape)
# ---------------------------------------------------------------------------


def test_regression_prior_club_regular_does_not_retain_start_probability(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, TRANSFER_BACKUP)
    rows = _rows(conn)
    subject, competitor = rows[60], rows[61]

    # 1. The incumbent who actually started outranks the prior-club regular.
    assert competitor["p_start"] > subject["p_start"]

    # 2. The three available non-selections are seen as evidence at all.
    assert subject["evidence_classes"][minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES] == 3
    assert subject["start_evidence"]["available_non_start_zero_minute_observations"] == 3

    # 3. The prior role is downweighted, explicitly and auditably.
    assert subject["prev_season_prior"]["effective_sample_size"] == pytest.approx(12.0 * 0.5)
    assert "role_discontinuity_prev_ess" in subject["prev_season_prior"]["ess_discounts"]
    assert subject["prev_season_prior"]["start_ess_after_discontinuity"] == pytest.approx(
        round(12.0 * 0.5 * 0.35, 4)
    )
    assert "ROLE_DISCONTINUITY_PRIOR_DISCOUNT" in subject["risk_flags"]
    assert "MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE" in subject["risk_flags"]
    assert "CURRENT_SEASON_ZERO_MINUTE_NON_START_EVIDENCE_STRONG" in subject["risk_flags"]

    # 4. Role confidence is exposed as LOW, and the basis is stated.
    assert subject["role_evidence"]["role_confidence"] == "LOW"
    assert subject["role_evidence"]["prior_role_discontinuity"] is True
    assert "prior_role_not_reproduced_at_current_club" in subject["role_evidence"]["uncertainty_reasons"]
    conn.close()


def test_available_non_start_evidence_lowers_p_start_materially(tmp_path):
    """The same subject, with and without the three non-selections."""

    with_evidence = connect_database(tmp_path / "with.db")
    without_evidence = connect_database(tmp_path / "without.db")
    _gk_world(with_evidence, TRANSFER_BACKUP)
    stripped = {
        60: {**TRANSFER_BACKUP[60], "pgw": []},
        61: TRANSFER_BACKUP[61],
    }
    _gk_world(without_evidence, stripped)

    with_rows = _rows(with_evidence)
    without_rows = _rows(without_evidence)
    assert with_rows[60]["p_start"] < without_rows[60]["p_start"]
    # Material, not incidental: at least a fifth of the raw probability.
    assert without_rows[60]["p_start"] - with_rows[60]["p_start"] >= 0.20 * without_rows[60]["p_start"]
    with_evidence.close()
    without_evidence.close()


def test_prior_alone_would_have_been_much_higher(tmp_path):
    """Shows the defect the taxonomy removes: prior-only would keep him a 'starter'."""

    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, TRANSFER_BACKUP)
    subject = _rows(conn)[60]
    prior_only = subject["start_evidence"]["prior_start_rate"]  # prior, pre-observation
    assert prior_only > 0.40
    assert subject["p_start"] < prior_only
    conn.close()


# ---------------------------------------------------------------------------
# Counter-tests — legitimate cases must not break
# ---------------------------------------------------------------------------


def test_counter_a_injured_regular_starter_is_not_demoted(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {"status": "a", "pgw": [(1, 1, 90, 1), (2, 2, 90, 1)], "prior": (35, 3150)},
            61: {
                "status": "i",
                "pgw": [_unused(1, 1), _unused(2, 2), _unused(3, 3)],
                "prior": (35, 3150),
            },
        },
    )
    rows = _rows(conn)
    injured = rows[61]
    assert injured["evidence_classes"][minutes_model.EVIDENCE_UNAVAILABLE] == 3
    assert injured["role_evidence"]["prior_role_discontinuity"] is False
    assert "ROLE_DISCONTINUITY_PRIOR_DISCOUNT" not in injured["risk_flags"]
    assert "MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE" not in injured["risk_flags"]
    # The injury absence must not be read as losing the role.
    assert "role_discontinuity_prev_ess" not in injured["prev_season_prior"]["ess_discounts"]
    conn.close()


def test_counter_b_bench_appearance_is_distinct_from_non_start(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {"status": "a", "pgw": [(1, 1, 20, 0), (2, 2, 25, 0), (3, 3, 15, 0)], "prior": (35, 3150)},
            61: {"status": "a", "pgw": [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 90, 1)]},
        },
    )
    cameo = _rows(conn)[60]
    assert cameo["evidence_classes"][minutes_model.EVIDENCE_BENCH_APPEARANCE] == 3
    assert cameo["evidence_classes"][minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES] == 0
    assert cameo["start_evidence"]["cameo_appearances"] == 3
    assert cameo["expected_minutes"] > 0.0
    conn.close()


def test_counter_c_return_from_suspension_is_not_role_loss(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {
                "status": "a",
                "pgw": [(1, 1, 90, 1), _unused(2, 2), _unused(3, 3)],
                "prior": (35, 3150),
            },
            61: {"status": "a", "pgw": [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 90, 1)]},
        },
    )
    # Snapshots: available GW1, suspended GW2-3, available again before GW4.
    with conn:
        conn.execute("UPDATE player_snapshots SET status='s' WHERE player_id=60")
        conn.execute(
            "UPDATE player_snapshots SET event_context=2 WHERE player_id=60 AND status='s'"
        )
    rows = _rows(conn)
    subject = rows[60]
    assert subject["evidence_classes"][minutes_model.EVIDENCE_UNAVAILABLE] == 2
    assert subject["evidence_classes"][minutes_model.EVIDENCE_STARTED] == 1
    assert subject["role_evidence"]["prior_role_discontinuity"] is False
    conn.close()


def test_counter_d_genuine_rotation_pair_stays_uncertain_not_conflicted(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {"status": "a", "pgw": [(1, 1, 90, 1), _unused(2, 2), (3, 3, 90, 1)], "prior": (20, 1800)},
            61: {"status": "a", "pgw": [_unused(1, 1), (2, 2, 90, 1), _unused(3, 3)], "prior": (20, 1800)},
        },
    )
    rows = _rows(conn)
    for pid in (60, 61):
        assert rows[pid]["role_evidence"]["prior_role_discontinuity"] is False
        assert "ROLE_DISCONTINUITY_PRIOR_DISCOUNT" not in rows[pid]["risk_flags"]
    # Neither is asserted as clearly the starter: they are close.
    assert abs(rows[60]["p_start"] - rows[61]["p_start"]) < 0.25
    conn.close()


def test_counter_e_new_transfer_without_current_matches_keeps_prior_but_lower_confidence(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {"status": "a", "pgw": [], "prior": (35, 3150)},
            61: {"status": "a", "pgw": [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 90, 1)]},
        },
    )
    rows = _rows(conn)
    newcomer = rows[60]
    # No current-team matches: no discontinuity is claimed, and the prior stands.
    assert newcomer["role_evidence"]["prior_role_discontinuity"] is False
    assert "ROLE_DISCONTINUITY_PRIOR_DISCOUNT" not in newcomer["risk_flags"]
    assert newcomer["role_evidence"]["p_start_basis"] == "prior_only"
    assert newcomer["start_evidence"]["prior_start_rate"] > 0.4
    # ...but confidence is not HIGH on a prior alone.
    assert newcomer["role_evidence"]["role_confidence"] in {"MEDIUM", "LOW"}
    conn.close()


def test_counter_f_single_emergency_benching_does_not_trigger_demotion(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {"status": "a", "pgw": [(1, 1, 90, 1), (2, 2, 90, 1), _unused(3, 3)], "prior": (35, 3150)},
            61: {"status": "a", "pgw": [_unused(1, 1), _unused(2, 2), _unused(3, 3)]},
        },
    )
    rows = _rows(conn)
    one_benching = rows[60]
    assert one_benching["evidence_classes"][minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES] == 1
    assert one_benching["role_evidence"]["prior_role_discontinuity"] is False
    assert "ROLE_DISCONTINUITY_PRIOR_DISCOUNT" not in one_benching["risk_flags"]
    assert "CURRENT_SEASON_ZERO_MINUTE_NON_START_EVIDENCE_STRONG" not in one_benching["risk_flags"]
    # Three unseated matches is what marks the backup, not one.
    assert rows[61]["role_evidence"]["prior_role_discontinuity"] is True or (
        rows[61]["evidence_classes"][minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES] == 3
    )
    conn.close()


# ---------------------------------------------------------------------------
# Diagnostics and downstream exposure
# ---------------------------------------------------------------------------


def test_role_diagnostics_are_exposed_for_downstream_confidence(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, TRANSFER_BACKUP)
    rows = _rows(conn)
    for pid in (60, 61):
        evidence = rows[pid]["role_evidence"]
        for key in (
            "prior_role_strength",
            "prior_role_discontinuity",
            "current_season_starts",
            "available_observations",
            "available_non_start_zero_minute_observations",
            "conflict_flags",
            "role_confidence",
            "uncertainty_reasons",
            "p_start_basis",
        ):
            assert key in evidence, (pid, key)
        assert rows[pid]["evidence_classes"]
    # The confident case and the conflicted case are distinguishable downstream.
    assert rows[61]["role_evidence"]["role_confidence"] == "HIGH"
    assert rows[60]["role_evidence"]["role_confidence"] == "LOW"
    conn.close()


def test_scouting_conflict_is_flagged_when_note_opposes_current_starts(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {
                "status": "a",
                "pgw": [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 90, 1)],
                "scout": ("rotation_risk", "high"),
            }
        },
    )
    row = _rows(conn)[60]
    assert "SCOUTING_ROLE_CONFLICT" in row["risk_flags"]
    assert "SCOUTING_ROLE_CONFLICT" in row["role_evidence"]["conflict_flags"]
    conn.close()


def test_start_rate_denominator_uses_only_attributable_rows(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(
        conn,
        {
            60: {
                "status": "i",
                "pgw": [(1, 1, 90, 1), (2, 2, 90, 1), _unused(3, 3)],
                "prior": (35, 3150),
            }
        },
    )
    with conn:
        conn.execute("UPDATE player_snapshots SET status='i' WHERE player_id=60")
        conn.execute("UPDATE player_snapshots SET event_context=3 WHERE player_id=60")
    row = _rows(conn)[60]
    evidence = row["start_evidence"]
    assert evidence["observed_rows"] == 3
    assert evidence["available_observation_rows"] == 2
    assert evidence["starts_observed"] == 2
    conn.close()


# ---------------------------------------------------------------------------
# Downstream decision confidence (objective unchanged)
# ---------------------------------------------------------------------------


def test_decision_confidence_flags_the_sep12_shape_as_fragile():
    from fpl_brain import decision_confidence as dc

    # The certified Sep-12 margin was 41.7291 vs 41.6336 = ~0.0955 CORE, and the
    # chosen goalkeeper's role evidence contradicted the model.
    subject_role = {
        "role_confidence": "LOW",
        "conflict_flags": [
            "MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE",
            "ROLE_DISCONTINUITY_PRIOR_DISCOUNT",
            "CURRENT_SEASON_ZERO_MINUTE_NON_START_EVIDENCE_STRONG",
        ],
        "prior_role_discontinuity": True,
        "uncertainty_reasons": ["prior_role_not_reproduced_at_current_club"],
    }
    # R4B.2c: near-tie now comes from the PAIRED CRN test, not a fixed CORE margin.
    result = dc.classify_decision_confidence(
        paired={"mean_difference": 0.0955, "paired_se": 0.10},
        role_evidence={497: subject_role},
        focus_player_ids=[497],
        selected_core=41.7291,
        alternative_cores={"start_raya": 41.6336},
    )
    assert result["state"] == dc.CONFIDENCE_MODEL_EVIDENCE_CONFLICT
    assert result["margin_core"] == pytest.approx(0.0955, abs=0.0001)
    assert result["near_tie"] is True
    assert result["role_conflicts"] and result["role_conflicts"][0]["player_id"] == 497
    assert result["affects_ranking"] is False


def test_decision_confidence_strong_when_clear_and_unconflicted():
    from fpl_brain import decision_confidence as dc

    # R4B.2c correction 2: a decisive recommendation REQUIRES the canonical paired
    # diagnostic; absent paired evidence is not "not near tied".
    result = dc.classify_decision_confidence(
        paired={"mean_difference": 5.0, "paired_se": 0.10},
        role_evidence={1: {"role_confidence": "HIGH", "conflict_flags": []}},
        focus_player_ids=[1],
    )
    assert result["state"] == dc.CONFIDENCE_STRONG
    assert result["near_tie"] is False

    # ...and the same clean evidence WITHOUT a paired record is INCOMPLETE, never STRONG
    missing = dc.classify_decision_confidence(
        paired=None,
        role_evidence={1: {"role_confidence": "HIGH", "conflict_flags": []}},
        focus_player_ids=[1],
    )
    assert missing["state"] == dc.CONFIDENCE_INCOMPLETE
    assert missing["decisive"] is False


def test_decision_confidence_reports_a_plain_near_tie():
    from fpl_brain import decision_confidence as dc

    # R4B.2c: a small CORE margin alone is NOT a near tie; the paired test decides.
    result = dc.classify_decision_confidence(
        paired={"mean_difference": 0.10, "paired_se": 0.20},
        role_evidence={1: {"role_confidence": "HIGH", "conflict_flags": []}},
        focus_player_ids=[1],
    )
    assert result["state"] == dc.CONFIDENCE_NEAR_TIE
    assert result["near_tie"] is True


def test_decision_confidence_downgrades_for_role_uncertainty():
    from fpl_brain import decision_confidence as dc

    result = dc.classify_decision_confidence(
        paired={"mean_difference": 0.10, "paired_se": 0.20},
        role_evidence={1: {"role_confidence": "LOW", "conflict_flags": [], "uncertainty_reasons": ["x"]}},
        focus_player_ids=[1],
    )
    assert result["state"] == dc.CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY
    assert result["near_tie"] is True
    dc.assert_confidence_invariants(result)

    # ...and with no near tie the same uncertainty is ROLE_UNCERTAINTY (R4B.2c)
    apart = dc.classify_decision_confidence(
        paired={"mean_difference": 5.0, "paired_se": 0.20},
        role_evidence={1: {"role_confidence": "LOW", "conflict_flags": [], "uncertainty_reasons": ["x"]}},
        focus_player_ids=[1],
    )
    assert apart["state"] == dc.CONFIDENCE_ROLE_UNCERTAINTY
    assert apart["near_tie"] is False
    dc.assert_confidence_invariants(apart)


def test_decision_confidence_ignores_non_starters():
    from fpl_brain import decision_confidence as dc

    result = dc.classify_decision_confidence(
        paired={"mean_difference": 5.0, "paired_se": 0.10},
        role_evidence={
            1: {"role_confidence": "HIGH", "conflict_flags": []},
            999: {"role_confidence": "LOW", "conflict_flags": ["MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE"]},
        },
        starter_ids=[1],
    )
    assert result["state"] == dc.CONFIDENCE_STRONG


# ---------------------------------------------------------------------------
# R2B.1 — no matchday-squad or bench claim without explicit evidence
# ---------------------------------------------------------------------------


def test_inferred_class_never_claims_bench_or_squad_membership():
    """The official payload cannot prove bench membership, so the inferred class
    must say only what is known: available, did not start, 0 minutes."""

    history = [
        {"id": 1, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": "a"},
        {"id": 2, "event_context": 4, "captured_at": "2026-09-10T00:00:00Z", "status": "a"},
    ]
    rows = [
        {"fixture_event": 3, "minutes": 0, "starts": 0},
        {"fixture_event": 4, "minutes": 0, "starts": 0},
    ]
    out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
    for row in out:
        assert row["evidence_class"] == minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES
        assert "matchday-squad membership unknown" in row["evidence_reason"]
        assert row["evidence_class"] not in minutes_model.EVIDENCE_EXPLICIT_SOURCE_ONLY


def test_explicit_source_classes_are_never_produced_by_inference():
    """Across every branch of the classifier, reserved classes stay unreachable."""

    histories = {
        "available": [{"id": 1, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": "a"}],
        "injured": [{"id": 1, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": "i"}],
        "suspended": [{"id": 1, "event_context": 3, "captured_at": "2026-09-03T00:00:00Z", "status": "s"}],
        "none": None,
    }
    rows = [
        {"fixture_event": 3, "minutes": 90, "starts": 1},
        {"fixture_event": 3, "minutes": 30, "starts": 0},
        {"fixture_event": 3, "minutes": 0, "starts": 0},
        {"fixture_event": 3, "minutes": None, "starts": None},
        {"fixture_event": 3, "minutes": 0, "starts": None},
    ]
    for label, history in histories.items():
        out = minutes_model.classify_evidence_rows(rows, snapshot_history=history)
        produced = {row["evidence_class"] for row in out}
        assert produced.isdisjoint(set(minutes_model.EVIDENCE_EXPLICIT_SOURCE_ONLY)), label
        assert minutes_model.EVIDENCE_NOT_IN_MATCHDAY_SQUAD not in produced, label


def test_payload_reports_squad_evidence_as_unknown_and_prior_club_unverified(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, TRANSFER_BACKUP)
    subject = _rows(conn)[60]
    evidence = subject["role_evidence"]
    assert evidence["matchday_squad_evidence"] == "UNKNOWN_NO_OFFICIAL_SQUAD_FIELD"
    assert evidence["prior_club_provenance"] == "UNVERIFIED_IN_STORED_EVIDENCE"
    assert "MATCHDAY_SQUAD_EVIDENCE_UNAVAILABLE" in subject["risk_flags"]
    # The discontinuity is labelled as a diagnostic, not as a proven transfer.
    assert "ROLE_DISCONTINUITY_PRIOR_DISCOUNT" in subject["risk_flags"]
    assert "PRIOR_CLUB_UNVERIFIED" in subject["risk_flags"]
    assert "TRANSFER_ROLE_CHANGE" not in subject["risk_flags"]
    assert evidence["role_discontinuity_diagnostics"] == [
        "PRIOR_CLUB_UNVERIFIED",
        "ROLE_DISCONTINUITY_PRIOR_DISCOUNT",
    ]
    conn.close()


# ---------------------------------------------------------------------------
# v1.8.0 — a non-reproduced prior role may not out-pull the evidence against it
# ---------------------------------------------------------------------------
# The GW5 defect: a strong previous-season prior role that had NOT reproduced was
# detected (MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE), published, and then ignored by
# the produced probability, because only the prior's effective sample size was
# discounted while its PULL in the start blend stayed at full strength.  A
# goalkeeper with zero current-season starts therefore kept a materially non-zero
# start probability, which the lineup resolver then legitimately exploited through
# goalkeeper autosub optionality.
#
# ``strong_zero_minute_non_start_evidence_threshold`` gates ONLY the strong
# evidence diagnostic and (now) the extra prior-pull discount, so raising it
# reproduces the pre-v1.8.0 probability on the repaired tree.  That is what makes
# these tests discriminating without hard-coding a probability.

NO_DISCOUNT_CONFIG = minutes_model.MinutesModelConfig(
    strong_zero_minute_non_start_evidence_threshold=10 ** 6
)


def _stale_prior_subject(conn, config=None):
    return _rows(conn, config)[60]


def test_a_non_reproduced_prior_role_no_longer_out_pulls_the_evidence(tmp_path):
    """The stale-prior GK's start probability must be materially reduced.

    The probability assertion comes FIRST deliberately: on the predecessor this is
    the assertion that fails, and it fails because the probability did not move --
    the intended defect -- rather than on a missing diagnostic field.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, TRANSFER_BACKUP)
    subject = _stale_prior_subject(conn)
    predecessor = _stale_prior_subject(conn, NO_DISCOUNT_CONFIG)

    # The discount is real and material: this is the defect, asserted against the
    # SAME fixture evaluated with the gate disabled, not against a magic number.
    assert subject["p_start"] < predecessor["p_start"]
    assert subject["p_start"] < 0.75 * predecessor["p_start"]

    cfg = minutes_model.MinutesModelConfig()
    assert subject["role_evidence"]["prior_pull_discount"] == pytest.approx(
        cfg.prev_season_role_discontinuity_ess_discount
    )
    assert subject["role_evidence"]["prior_pull_discount_basis"] == "ROLE_DISCONTINUITY_UNREPRODUCED_PRIOR"
    assert subject["role_evidence"]["fresh_role_change_evidence"] is False
    conn.close()


def test_the_discount_is_a_downweight_never_a_ban(tmp_path):
    """A contradicted prior still leaves the player selectable."""

    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, TRANSFER_BACKUP)
    subject = _stale_prior_subject(conn)
    assert subject["p_start"] > 0.0
    assert subject["expected_minutes"] > 0.0
    assert subject["p_zero"] < 1.0
    conn.close()


def test_fresh_role_change_evidence_leaves_the_probability_untouched(tmp_path):
    """COUNTEREXAMPLE: a genuine role change is never banned or forced to zero.

    The same zero-current-season GK, now carrying canonical fresh evidence of a
    role transition (a scouting role confirmation).  The extra discount must not
    apply, and the estimate must be exactly what it was before this repair.
    """

    world = {
        60: {
            **TRANSFER_BACKUP[60],
            "scout": ("role_security_5gw", "very_high"),
        },
        61: TRANSFER_BACKUP[61],
    }
    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, world)
    with_evidence = _stale_prior_subject(conn)
    predecessor = _stale_prior_subject(conn, NO_DISCOUNT_CONFIG)

    assert "START_ROLE_CONFIRMED" in with_evidence["risk_flags"]
    assert with_evidence["role_evidence"]["fresh_role_change_evidence"] is True
    assert with_evidence["role_evidence"]["prior_pull_discount"] == pytest.approx(1.0)
    assert with_evidence["role_evidence"]["prior_pull_discount_basis"] == "FRESH_ROLE_CHANGE_EVIDENCE"
    # Untouched, and strictly selectable.
    assert with_evidence["p_start"] == pytest.approx(predecessor["p_start"])
    assert with_evidence["p_start"] > 0.0
    conn.close()


def test_the_current_starter_is_never_touched_by_the_discount(tmp_path):
    """A player with real current-season starts never reaches this branch."""

    conn = connect_database(tmp_path / "fpl.db")
    _gk_world(conn, TRANSFER_BACKUP)
    rows = _rows(conn)
    competitor = rows[61]
    assert competitor["role_evidence"]["prior_pull_discount"] == pytest.approx(1.0)
    assert competitor["role_evidence"]["prior_pull_discount_basis"] == "NO_ROLE_CONFLICT"
    assert competitor["p_start"] > rows[60]["p_start"]
    conn.close()
