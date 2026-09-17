"""PE-2E evaluation-contract closures: the clean-sheet target and per-event run identity.

Two concrete predecessor defects, each reproduced from source and measurement before
any code changed:

PE2E-P2-01  ``calibration.evaluate_monte_carlo_run`` scored the simulator's
            ``p_clean_sheet_eligible`` against the raw ``clean_sheets`` column, but
            that probability's flag is set in the same block that AWARDS clean-sheet
            points and requires ``clean_sheet_points_for(position) > 0``.  Measured on
            the nine Monte Carlo runs with played fixtures: 0 forwards ever have a
            non-zero probability, yet 81 forward rows carry ``clean_sheets = 1``, so
            81 observations scored 0.0 against a realised 1.0.

PE2E-P2-02  ``walk_forward.EvaluationIdentity.as_dict()`` built its run map as
            ``{name: run_id for name, run_id in ...}``, which keeps only the LAST run
            per family.  Four per-event xPts runs serialised as ``{"xpts_v1": 419}``.

Every expectation here is hand-derived, and the predecessor's value is quoted beside
the repaired one so the two can be compared directly.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from fpl_brain import analytics
from fpl_brain import calibration
from fpl_brain import monte_carlo as mc
from fpl_brain import repositories as repo
from fpl_brain import walk_forward as wf
from fpl_brain import walk_forward_scoreboard as sb
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PositionRecord,
    TeamRecord,
)
from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES as RULES, ScoringRules

CUTOFF = "2026-09-10T12:00:00Z"
CODE_SNAPSHOT = "sha256:" + "a" * 64
EVENT = 4
FIXTURE = 100

#: player -> (element_type, minutes, goals_conceded, clean_sheets, p_clean_sheet_eligible)
MC_ROWS = {
    10: (2, 90, 0, 1, 0.6),   # DEF, realised clean sheet
    11: (3, 90, 0, 1, 0.5),   # MID, realised clean sheet (MID earns 1 point)
    12: (4, 90, 0, 1, 0.0),   # FWD: raw flag set, but he scores NOTHING for it
    13: (1, 45, 0, 0, 0.2),   # GKP, under the 60-minute boundary
    14: (2, 90, 1, 0, 0.3),   # DEF, conceded
}


def _players(conn) -> None:
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_positions(
            conn,
            [PositionRecord(id=p, singular_name_short=n)
             for p, n in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))],
        )
        repo.upsert_players(
            conn,
            [PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}",
                          team_id=1, element_type=elements)
             for pid, (elements, *_rest) in MC_ROWS.items()],
        )


def _mc_world(conn):
    """A finalised event with one played fixture and five position-diverse rows."""

    _players(conn)
    with conn:
        repo.upsert_events(conn, [EventRecord(id=EVENT, finished=1, data_checked=1,
                                             deadline_time="2026-09-05T17:30:00Z", raw_json={})])
        repo.upsert_fixtures(conn, [FixtureRecord(
            id=FIXTURE, event=EVENT, team_h=1, team_a=2, team_h_score=0, team_a_score=0,
            started=1, finished=1, finished_provisional=0, minutes=90,
        )])
        for player_id, (_elements, minutes, conceded, clean_sheets, _p) in sorted(MC_ROWS.items()):
            repo.upsert_player_gameweeks(conn, [PlayerGameweekRecord(
                player_id=player_id, event=EVENT, fixture_id=FIXTURE, opponent_team=2, was_home=1,
                minutes=minutes, starts=1 if minutes >= 60 else 0, total_points=0,
                goals_scored=0, assists=0, clean_sheets=clean_sheets, goals_conceded=conceded,
                saves=0, bonus=0, yellow_cards=0, defensive_contribution=0,
                source="element_summary",
            )])
    run_id = analytics.create_projection_run(
        conn, model_family=mc.MONTE_CARLO_MODEL_FAMILY, model_version=mc.MONTE_CARLO_MODEL_VERSION,
        planning_event=EVENT, planning_context_hash="hash", data_cutoff=CUTOFF,
        scouting_cutoff=None, official_run_ids={}, deadline_status="PRE_DEADLINE",
    )
    with conn:
        for player_id, (_elements, _minutes, _conceded, _cs, probability) in sorted(MC_ROWS.items()):
            analytics.freeze_monte_carlo_distribution(
                conn, run_id, player_id=player_id, fixture_id=FIXTURE, event=EVENT,
                team_id=1, opponent_id=2, position="MID", xpts_run_id=1, minutes_run_id=1,
                team_run_id=1, rate_run_id=1,
                payload={
                    "mean_core": 2.0, "mean_total_proxy": 2.0,
                    "q10": 1.0, "q25": 1.0, "q50": 2.0, "q75": 3.0, "q90": 4.0,
                    "p_goal": 0.1, "p_assist": 0.1,
                    "p_clean_sheet_eligible": probability, "p_defcon_hit": 0.2,
                },
                model_version=mc.MONTE_CARLO_MODEL_VERSION,
            )
    analytics.finish_projection_run(conn, run_id, "complete")
    return run_id


def _xpts_world(conn, *, clean_sheet_probability: float):
    """One FWD xPts row whose raw clean-sheet flag is set."""

    _players(conn)
    with conn:
        repo.upsert_events(conn, [EventRecord(id=EVENT, finished=1, data_checked=1,
                                             deadline_time="2026-09-05T17:30:00Z", raw_json={})])
        repo.upsert_fixtures(conn, [FixtureRecord(
            id=FIXTURE, event=EVENT, team_h=1, team_a=2, team_h_score=1, team_a_score=0,
            started=1, finished=1, finished_provisional=0, minutes=90,
        )])
        repo.upsert_player_gameweeks(conn, [PlayerGameweekRecord(
            player_id=12, event=EVENT, fixture_id=FIXTURE, opponent_team=2, was_home=1,
            minutes=90, starts=1, total_points=6, goals_scored=1, assists=0,
            clean_sheets=1, goals_conceded=0, saves=0, bonus=0, yellow_cards=0,
            defensive_contribution=0, source="element_summary",
        )])
    run_id = analytics.create_projection_run(
        conn, model_family="xpts_v1", model_version="xpts_v1.4.1", planning_event=EVENT,
        planning_context_hash="hash", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )
    with conn:
        conn.execute(
            "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id,"
            " event, team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id,"
            " payload_json, model_version, scoring_rules_version, generated_at)"
            " VALUES (?,12,?,?,1,2,'FWD',1,1,1,?,'xpts_v1.4.1','fpl_scoring_2026_27_v1.0.0',?)",
            (run_id, FIXTURE, EVENT,
             json.dumps({"total_xpts": 6.0, "clean_sheet_probability": clean_sheet_probability,
                         "clean_sheet_xpts": 0.0}),
             CUTOFF),
        )
    analytics.finish_projection_run(conn, run_id, "complete")
    return run_id


# ---------------------------------------------------------------------------
# The canonical helper
# ---------------------------------------------------------------------------


def test_canonical_clean_sheet_eligibility_requires_all_three_clauses():
    """Hand-derived truth table over the three clauses of the scoring event."""

    # (position, minutes, goals_conceded) -> did the clean-sheet SCORING event occur
    assert RULES.earns_clean_sheet_points("GKP", 90, 0) is True
    assert RULES.earns_clean_sheet_points("DEF", 90, 0) is True
    assert RULES.earns_clean_sheet_points("MID", 90, 0) is True
    # A forward is on for 90 minutes of a clean sheet and is awarded nothing.
    assert RULES.earns_clean_sheet_points("FWD", 90, 0) is False
    # The 60-minute boundary.
    assert RULES.earns_clean_sheet_points("DEF", 60, 0) is True
    assert RULES.earns_clean_sheet_points("DEF", 59, 0) is False
    # Conceding disqualifies the sheet.
    assert RULES.earns_clean_sheet_points("DEF", 90, 1) is False
    # A zero-minute genuine non-appearance never earns a clean sheet.
    assert RULES.earns_clean_sheet_points("DEF", 0, 0) is False


def test_the_helper_reads_its_clauses_from_the_rule_object_not_from_constants():
    """A custom rule set must change the answer, so the clauses are not hard-coded."""

    custom = ScoringRules(
        clean_sheet_points={"GKP": 4, "DEF": 4, "MID": 1, "FWD": 2},
        clean_sheet_minutes_required=30,
    )
    # A forward now scores for a clean sheet, and the boundary moved.
    assert custom.earns_clean_sheet_points("FWD", 90, 0) is True
    assert custom.earns_clean_sheet_points("DEF", 30, 0) is True
    assert custom.earns_clean_sheet_points("DEF", 29, 0) is False
    assert RULES.earns_clean_sheet_points("FWD", 90, 0) is False, "the default rules were mutated"
    # The helper is additive: it must not have changed the hashed scoring values.
    assert RULES.scoring_hash() == (
        "sha256:d432151937ce5616d9ebb659a0deb8e5135d6aed3169a196420c367aa1eceb4a"
    )


def test_a_forward_clean_sheet_is_not_a_clean_sheet_scoring_event():
    """The exact case the predecessor scored wrongly."""

    minutes, conceded = 90, 0
    assert RULES.clean_sheet_points_for("FWD") == 0
    assert RULES.earns_clean_sheet_points("FWD", minutes, conceded) is False
    # For the positions that DO score, the scoring event and the raw flag coincide.
    for position in ("GKP", "DEF", "MID"):
        assert RULES.earns_clean_sheet_points(position, minutes, conceded) is True
        assert RULES.clean_sheet_points_for(position) > 0


# ---------------------------------------------------------------------------
# P2-01: the Monte Carlo calibration target
# ---------------------------------------------------------------------------


def test_the_mc_clean_sheet_brier_target_matches_the_producer(tmp_path):
    """The simulator's flag means "earned clean-sheet points", so the target must too.

    Squared errors with the repaired target:
        DEF 90' clean sheet   (0.6 - 1.0)^2 = 0.16
        MID 90' clean sheet   (0.5 - 1.0)^2 = 0.25
        FWD 90' raw flag set  (0.0 - 0.0)^2 = 0.00   <-- the predecessor used 1.0 here
        GKP 45' no clean sh.  (0.2 - 0.0)^2 = 0.04
        DEF 90' conceded      (0.3 - 0.0)^2 = 0.09
    sum = 0.54 over 5 rows -> 0.108.
    The predecessor's raw-flag target summed to 1.54 -> 0.308.
    """

    conn = connect_database(tmp_path / "mc.db")
    try:
        run_id = _mc_world(conn)
        metrics = calibration.evaluate_monte_carlo_run(conn, run_id, EVENT)
        assert metrics["brier_clean_sheet"] == pytest.approx(0.108, abs=1e-9)
        assert metrics["brier_clean_sheet"] != pytest.approx(0.308, abs=1e-9), (
            "the raw clean_sheets flag was used as the target"
        )
    finally:
        conn.close()


def test_the_mc_forward_row_contributes_a_zero_squared_error(tmp_path):
    """Isolated: the ONE row whose target changed must now agree perfectly."""

    conn = connect_database(tmp_path / "mc2.db")
    try:
        run_id = _mc_world(conn)
        metrics = calibration.evaluate_monte_carlo_run(conn, run_id, EVENT)
        # 0.54 is the repaired sum; the only row that moved is the forward's, from
        # 1.00 to 0.00, i.e. a difference of exactly 1.00 over 5 rows = 0.2.
        assert metrics["brier_clean_sheet"] == pytest.approx(0.54 / 5, abs=1e-9)
        assert (0.308 - metrics["brier_clean_sheet"]) == pytest.approx(1.0 / 5, abs=1e-9)
    finally:
        conn.close()


def test_the_valid_positional_mc_targets_are_unchanged(tmp_path):
    """GKP/DEF/MID rows must produce exactly what the raw flag produced.

    Computed here from the raw flag independently, so a regression that moved these
    rows would show up as a difference rather than being absorbed.
    """

    conn = connect_database(tmp_path / "mc3.db")
    try:
        run_id = _mc_world(conn)
        metrics = calibration.evaluate_monte_carlo_run(conn, run_id, EVENT)
        from_raw_flag = []
        for player_id, (elements, _minutes, _conceded, clean_sheets, probability) in sorted(MC_ROWS.items()):
            if elements == 4:      # the forward is the only position that changed
                continue
            from_raw_flag.append((probability - (1.0 if clean_sheets else 0.0)) ** 2)
        # 0.16 + 0.25 + 0.04 + 0.09 over the four non-forward rows.
        assert sum(from_raw_flag) == pytest.approx(0.54, abs=1e-9)
        # Zero total squared error is contributed by the forward under the repair, so
        # the aggregate equals the non-forward subtotal.
        assert metrics["brier_clean_sheet"] == pytest.approx(sum(from_raw_flag) / 5, abs=1e-9)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P2-01: the xPts target deliberately stays on the physical flag
# ---------------------------------------------------------------------------


def test_the_xpts_clean_sheet_target_stays_on_the_physical_flag(tmp_path):
    """A different persisted probability with a different meaning keeps its own target.

    ``xpts._clean_sheet_probability`` models "60+ minutes of exposure with no
    opponent goal during it" and is stored for every position -- measured live, 477
    forward rows carry a non-zero value -- while only the expected-POINTS layer
    multiplies it by ``clean_sheet_points_for``.  Its realised target is therefore
    the official physical flag, and harmonising the two targets would break this one.
    """

    conn = connect_database(tmp_path / "xpts.db")
    try:
        run_id = _xpts_world(conn, clean_sheet_probability=0.3)
        metrics = calibration.evaluate_xpts_run(conn, run_id, EVENT)
        # (0.3 - 1.0)^2 = 0.49 against the physical flag.
        assert metrics["clean_sheet_brier"] == pytest.approx(0.49, abs=1e-9)
        # The position-conditioned target would have given (0.3 - 0.0)^2 = 0.09.
        assert metrics["clean_sheet_brier"] != pytest.approx(0.09, abs=1e-9)
    finally:
        conn.close()


def test_the_two_clean_sheet_probabilities_are_distinct_objects(tmp_path):
    """Different persisted fields, different producers, therefore different targets."""

    conn = connect_database(tmp_path / "both.db")
    try:
        mc_run = _mc_world(conn)
        # The forward's simulator probability is zero: he is excluded from the event.
        forward = [
            row for row in analytics.monte_carlo_distributions(conn, mc_run)
            if int(row["player_id"]) == 12
        ]
        assert (forward[0]["payload"] or {})["p_clean_sheet_eligible"] == 0.0
        # The xPts field for the same player is a physical exposure measure and is not
        # forced to zero by his position.
        xpts_run = _xpts_world(conn, clean_sheet_probability=0.3)
        row = [r for r in analytics.xpts_projections(conn, xpts_run) if int(r["player_id"]) == 12][0]
        assert row["payload"]["clean_sheet_probability"] == pytest.approx(0.3)
        assert row["payload"]["clean_sheet_xpts"] == 0.0, "the points layer is conditioned separately"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P2-01: one definition shared by both consumers
# ---------------------------------------------------------------------------


def test_the_scoring_helper_is_the_single_definition_for_both_consumers(tmp_path, monkeypatch):
    """Neutralise the position clause ONCE and require BOTH consumers to move.

    If either the calibration evaluator or the walk-forward scoreboard carried its
    own copy of the rule, only one of the two would change.  Both changing is the
    proof that they delegate to one canonical helper.
    """

    conn = connect_database(tmp_path / "shared.db")
    try:
        mc_run = _mc_world(conn)
        before_calibration = calibration.evaluate_monte_carlo_run(conn, mc_run, EVENT)["brier_clean_sheet"]

        monkeypatch.setattr(
            ScoringRules,
            "earns_clean_sheet_points",
            lambda self, position, minutes, goals_conceded: (
                int(minutes) >= self.clean_sheet_minutes_required and int(goals_conceded) == 0
            ),
        )
        after_calibration = calibration.evaluate_monte_carlo_run(conn, mc_run, EVENT)["brier_clean_sheet"]
        # Without the position clause the forward's target becomes 1.0 -> +1.0/5.
        assert after_calibration == pytest.approx(before_calibration + 1.0 / 5, abs=1e-9)

        # The scoreboard's target is computed by the same helper.
        from fpl_brain import walk_forward_scoreboard as scoreboard

        assert (
            scoreboard._realised_probability_outcome(
                "clean_sheet_points", {"minutes": 90, "goals_conceded": 0}, "FWD"
            )
            == 1.0
        ), "the scoreboard does not delegate to the canonical helper"
    finally:
        conn.close()


def test_the_scoreboard_clean_sheet_target_is_zero_for_a_forward():
    """The walk-forward target for the scoring event, across positions."""

    from fpl_brain import walk_forward_scoreboard as scoreboard

    outcome = {"minutes": 90, "goals_conceded": 0}
    assert scoreboard._realised_probability_outcome("clean_sheet_points", outcome, "DEF") == 1.0
    assert scoreboard._realised_probability_outcome("clean_sheet_points", outcome, "MID") == 1.0
    assert scoreboard._realised_probability_outcome("clean_sheet_points", outcome, "GKP") == 1.0
    assert scoreboard._realised_probability_outcome("clean_sheet_points", outcome, "FWD") == 0.0
    assert scoreboard._realised_probability_outcome(
        "clean_sheet_points", {"minutes": 59, "goals_conceded": 0}, "DEF"
    ) == 0.0
    assert scoreboard._realised_probability_outcome(
        "clean_sheet_points", {"minutes": 90, "goals_conceded": 1}, "DEF"
    ) == 0.0
    # A missing realised column stays a gap and never becomes a zero.
    assert scoreboard._realised_probability_outcome(
        "clean_sheet_points", {"minutes": 90, "goals_conceded": None}, "DEF"
    ) is None


def test_calibration_and_the_scoreboard_agree_on_the_target_for_every_position():
    """The two consumers must return the SAME target for the same realised row."""

    from fpl_brain import walk_forward_scoreboard as scoreboard

    for position in ("GKP", "DEF", "MID", "FWD"):
        for minutes in (0, 30, 60, 90):
            for conceded in (0, 1, 3):
                expected = (
                    1.0
                    if RULES.earns_clean_sheet_points(position, minutes, conceded)
                    else 0.0
                )
                assert scoreboard._realised_probability_outcome(
                    "clean_sheet_points", {"minutes": minutes, "goals_conceded": conceded}, position
                ) == expected


# ---------------------------------------------------------------------------
# P2-02: multi-event run identity
# ---------------------------------------------------------------------------

EVENT_RUNS = {5: 380, 6: 393, 7: 406, 8: 419}
EVENT_BASELINE = {5: 369, 6: 382, 7: 395, 8: 408}
MULTI_EVENT_FIXTURES = {5: 100, 6: 200, 7: 300, 8: 400}


def _identity(per_event_runs, *, target_events=(5, 6, 7, 8), digest="sha256:" + "d" * 64):
    return wf.EvaluationIdentity(
        evaluation_version=wf.WALK_FORWARD_VERSION,
        grain=wf.GRAIN_PLAYER_EVENT,
        target_events=tuple(target_events),
        planning_cutoff=CUTOFF,
        per_event_runs=tuple(per_event_runs),
        baseline_kinds=(analytics.RECENT_POINTS_KIND,),
        eligible_population_digest=digest,
        model_versions=(("xpts_v1", "xpts_v1.4.1"),),
        code_snapshot_sha256=CODE_SNAPSHOT,
        code_revision="rev",
        missing_data_policy_version=wf.MISSING_DATA_POLICY_VERSION,
        outcome_state=tuple((e, "SCHEDULED") for e in target_events),
    )


def _four_event_runs():
    runs = []
    for event in sorted(EVENT_RUNS):
        runs.append((event, "xpts_v1", EVENT_RUNS[event]))
        runs.append((event, wf.BASELINE_FAMILY, EVENT_BASELINE[event]))
    return runs


def _serialised_run_ids(identity) -> set[int]:
    """Every run id a serialised identity carries, for EITHER representation.

    Reads only the run-bearing keys, so an event number can never be mistaken for a
    run number.  Tolerant across both shapes on purpose: this has to read the
    predecessor's family-keyed ``projection_run_ids`` / ``baseline_run_ids`` maps AND
    the repaired ``per_event_runs`` map, so that it can demonstrate the collapse on
    the revision that had it instead of failing on a missing attribute.
    """

    data = identity.as_dict()
    found: set[int] = set()

    def absorb(value) -> None:
        if isinstance(value, dict):
            for item in value.values():
                absorb(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                absorb(item)
        elif isinstance(value, int):
            found.add(value)

    for key in ("per_event_runs", "projection_run_ids", "baseline_run_ids"):
        if key in data:
            absorb(data[key])
    return found


def test_a_multi_event_identity_preserves_every_event_run():
    """A/B/C/D each appear once, in their own event. The predecessor kept only D."""

    serialised = _identity(_four_event_runs()).as_dict()["per_event_runs"]
    assert serialised == {
        "5": {wf.BASELINE_FAMILY: 369, "xpts_v1": 380},
        "6": {wf.BASELINE_FAMILY: 382, "xpts_v1": 393},
        "7": {wf.BASELINE_FAMILY: 395, "xpts_v1": 406},
        "8": {wf.BASELINE_FAMILY: 408, "xpts_v1": 419},
    }
    # The predecessor's output: a family-keyed map that kept only the last run.
    collapsed = {name: run for _event, name, run in _four_event_runs()}
    assert collapsed == {"xpts_v1": 419, wf.BASELINE_FAMILY: 408}
    assert serialised != collapsed

    flat = json.dumps(serialised)
    for run_id in EVENT_RUNS.values():
        assert flat.count(str(run_id)) == 1, f"run {run_id} is not present exactly once"
    for run_id in EVENT_BASELINE.values():
        assert flat.count(str(run_id)) == 1, f"baseline run {run_id} is not present exactly once"


def test_the_identity_exposes_lossless_per_event_accessors():
    identity = _identity(_four_event_runs())
    assert identity.runs_for_event(7) == {"xpts_v1": 406, wf.BASELINE_FAMILY: 395}
    assert identity.run_id_for(8, "xpts_v1") == 419
    assert identity.run_id_for(8, "defcon_v9") is None
    assert identity.runs_for_event(99) == {}


def test_reordering_the_input_map_does_not_change_the_canonical_output():
    forward = _four_event_runs()
    shuffled = [
        (8, wf.BASELINE_FAMILY, 408), (5, "xpts_v1", 380), (7, wf.BASELINE_FAMILY, 395),
        (6, "xpts_v1", 393), (5, wf.BASELINE_FAMILY, 369), (8, "xpts_v1", 419),
        (7, "xpts_v1", 406), (6, wf.BASELINE_FAMILY, 382),
    ]
    assert shuffled != forward, "the shuffle must actually differ"
    assert (
        json.dumps(_identity(forward).as_dict(), sort_keys=True)
        == json.dumps(_identity(shuffled).as_dict(), sort_keys=True)
    )
    assert sb.scoreboard_digest(_identity(forward).as_dict()) == sb.scoreboard_digest(
        _identity(shuffled).as_dict()
    )


def test_changing_one_events_run_changes_the_identity_digest():
    original = _identity(_four_event_runs())
    changed_runs = [
        (event, family, 999 if (event == 7 and family == "xpts_v1") else run)
        for event, family, run in _four_event_runs()
    ]
    changed = _identity(changed_runs)
    assert json.dumps(original.as_dict(), sort_keys=True) != json.dumps(changed.as_dict(), sort_keys=True)
    assert sb.scoreboard_digest(original.as_dict()) != sb.scoreboard_digest(changed.as_dict())
    # Only event 7 moved.
    assert changed.runs_for_event(7)["xpts_v1"] == 999
    assert changed.runs_for_event(6)["xpts_v1"] == 393


def test_a_single_event_identity_keeps_its_one_run_and_invents_no_events():
    identity = _identity([(5, "xpts_v1", 380), (5, wf.BASELINE_FAMILY, 369)], target_events=(5,))
    serialised = identity.as_dict()["per_event_runs"]
    assert serialised == {"5": {wf.BASELINE_FAMILY: 369, "xpts_v1": 380}}
    assert list(serialised) == ["5"], "no event may be fabricated"
    assert identity.runs_for_event(5)["xpts_v1"] == 380
    assert identity.as_dict()["target_events"] == [5]


def test_a_single_event_identity_loses_no_run_relative_to_the_flat_form():
    """Compatibility: the repaired shape still carries every run the flat form did."""

    pairs = [(5, "xpts_v1", 380), (5, wf.BASELINE_FAMILY, 369)]
    serialised = _identity(pairs, target_events=(5,)).as_dict()["per_event_runs"]
    recovered = {
        (int(event), family, run)
        for event, families in serialised.items()
        for family, run in families.items()
    }
    assert recovered == {(5, "xpts_v1", 380), (5, wf.BASELINE_FAMILY, 369)}


# ---------------------------------------------------------------------------
# P2-02: end to end through a real multi-event population build
# ---------------------------------------------------------------------------


def _multi_event_world(conn):
    """Four finalised events, each with its OWN xPts run and its own baseline run."""

    _players(conn)
    with conn:
        repo.upsert_events(conn, [
            EventRecord(id=event, finished=1, data_checked=1,
                        deadline_time=f"2026-09-{4 + event:02d}T17:30:00Z", raw_json={})
            for event in sorted(MULTI_EVENT_FIXTURES)
        ])
        repo.upsert_fixtures(conn, [
            FixtureRecord(id=fixture, event=event, team_h=1, team_a=2, finished=1, started=1,
                          kickoff_time=f"2026-09-{4 + event:02d}T14:00:00Z", raw_json={})
            for event, fixture in sorted(MULTI_EVENT_FIXTURES.items())
        ])
        for event, fixture in sorted(MULTI_EVENT_FIXTURES.items()):
            for player_id in sorted(MC_ROWS):
                conn.execute(
                    "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts,"
                    " total_points, goals_scored, assists, clean_sheets, goals_conceded, saves,"
                    " yellow_cards, defensive_contribution, source, updated_at, raw_json)"
                    " VALUES (?,?,?,90,1,5,0,0,1,0,0,0,0,'element_summary','2026-09-06T09:00:00Z','{}')",
                    (player_id, event, fixture),
                )

    runs: dict[str, dict[int, int]] = {"xpts_v1": {}, "baseline": {}}
    for event, fixture in sorted(MULTI_EVENT_FIXTURES.items()):
        xpts_run = analytics.create_projection_run(
            conn, model_family="xpts_v1", model_version="xpts_v1.4.1", planning_event=event,
            planning_context_hash=f"ctx{event}", data_cutoff=CUTOFF, scouting_cutoff=None,
            official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
        )
        baseline_run = analytics.create_projection_run(
            conn, model_family="baseline", model_version=analytics.BASELINE_MODEL_VERSION,
            planning_event=event, planning_context_hash=f"ctx{event}", data_cutoff=CUTOFF,
            scouting_cutoff=None, official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
        )
        with conn:
            for player_id in sorted(MC_ROWS):
                conn.execute(
                    "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id,"
                    " fixture_id, event, team_id, opponent_id, position, minutes_run_id, team_run_id,"
                    " rate_run_id, payload_json, model_version, scoring_rules_version, generated_at)"
                    " VALUES (?,?,?,?,1,2,'MID',1,1,1,?,'xpts_v1.4.1',"
                    "'fpl_scoring_2026_27_v1.0.0','2026-09-10T12:00:00Z')",
                    (xpts_run, player_id, fixture, event,
                     json.dumps({"total_xpts": 4.0, "p_start": 0.8, "p_60_plus": 0.7,
                                 "clean_sheet_probability": 0.5, "defcon_p_hit": 0.4})),
                )
                conn.execute(
                    "INSERT INTO frozen_predictions(projection_run_id, kind, player_id, fixture_id,"
                    " event, payload_json, model_version, generated_at)"
                    " VALUES (?,?,?,NULL,?,?,?,'2026-09-10T12:00:00Z')",
                    (baseline_run, analytics.RECENT_POINTS_KIND, player_id, event,
                     json.dumps({"value": 3.0}), analytics.BASELINE_MODEL_VERSION),
                )
        with conn:
            for run_id in (xpts_run, baseline_run):
                conn.execute("UPDATE projection_runs SET status='complete' WHERE id=?", (run_id,))
        runs["xpts_v1"][event] = xpts_run
        runs["baseline"][event] = baseline_run

    artifact = {
        "four_gw_certification_identity": "sha256:" + "e" * 64,
        "planning_cutoff": CUTOFF,
        "certified_bundles": {
            str(event): {
                "event": event,
                "cutoff": CUTOFF,
                "runs": {"xpts_v1": runs["xpts_v1"][event]},
                "model_versions": {"xpts_v1": "xpts_v1.4.1"},
            }
            for event in sorted(MULTI_EVENT_FIXTURES)
        },
    }
    return runs, artifact


def test_a_real_multi_event_build_preserves_every_events_run(tmp_path):
    """THE discriminating regression: four events, four runs, all four must survive.

    Written against the serialised value's CONTENT rather than its shape, so it
    fails on the predecessor for the intended reason -- the collapsed map keeps one
    run out of four -- instead of failing on a missing attribute and telling us
    nothing about which run was lost.
    """

    conn = connect_database(tmp_path / "multi.db")
    try:
        runs, artifact = _multi_event_world(conn)
        anchor = wf.discover_certified_anchor(conn, artifact, events=sorted(MULTI_EVENT_FIXTURES))
        population = wf.build_event_population(
            conn, anchor=anchor, events=sorted(MULTI_EVENT_FIXTURES), baseline_kind=None
        )
        mentioned = _serialised_run_ids(population.identity)
        for event, run_id in sorted(runs["xpts_v1"].items()):
            assert run_id in mentioned, (
                f"event {event}'s xPts run {run_id} is MISSING from the serialised identity "
                f"(it carries {sorted(mentioned)})"
            )
        assert len({*runs["xpts_v1"].values()}) == 4, "the four events must have distinct runs"
    finally:
        conn.close()


def test_a_real_multi_event_build_records_both_events_runs(tmp_path):
    """End to end, in the repaired shape: every event keeps its own run."""

    conn = connect_database(tmp_path / "two.db")
    try:
        runs, artifact = _multi_event_world(conn)
        anchor = wf.discover_certified_anchor(conn, artifact, events=sorted(MULTI_EVENT_FIXTURES))
        population = wf.build_event_population(
            conn, anchor=anchor, events=sorted(MULTI_EVENT_FIXTURES), baseline_kind=None
        )
        recorded = population.identity.as_dict()["per_event_runs"]
        assert recorded == {
            str(event): {"xpts_v1": runs["xpts_v1"][event]} for event in sorted(MULTI_EVENT_FIXTURES)
        }
        assert population.identity.run_id_for(6, "xpts_v1") == runs["xpts_v1"][6]
        assert population.identity.run_id_for(7, "defcon_v9") is None
    finally:
        conn.close()


def test_the_scoreboard_artifact_changes_when_one_events_run_changes(tmp_path):
    """Downstream proof: swapping one event's xPts run moves the artifact digest."""

    conn = connect_database(tmp_path / "two_digest.db")
    try:
        events = sorted(MULTI_EVENT_FIXTURES)
        runs, artifact = _multi_event_world(conn)
        first = sb.build_scoreboard(conn, artifact=artifact, events=events, top_k=2)
        assert first["identity"]["per_event_runs"]["6"]["xpts_v1"] == runs["xpts_v1"][6]

        # Point event 6 at a DIFFERENT, complete xPts run over the SAME fixture.
        with conn:
            replacement = analytics.create_projection_run(
                conn, model_family="xpts_v1", model_version="xpts_v1.4.1", planning_event=6,
                planning_context_hash="other", data_cutoff=CUTOFF, scouting_cutoff=None,
                official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
            )
            for player_id in sorted(MC_ROWS):
                conn.execute(
                    "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id,"
                    " fixture_id, event, team_id, opponent_id, position, minutes_run_id, team_run_id,"
                    " rate_run_id, payload_json, model_version, scoring_rules_version, generated_at)"
                    " VALUES (?,?,?,6,1,2,'MID',1,1,1,?,'xpts_v1.4.1',"
                    "'fpl_scoring_2026_27_v1.0.0','2026-09-10T12:00:00Z')",
                    (replacement, player_id, MULTI_EVENT_FIXTURES[6],
                     json.dumps({"total_xpts": 5.0, "p_start": 0.8, "p_60_plus": 0.7,
                                 "clean_sheet_probability": 0.5, "defcon_p_hit": 0.4})),
                )
            conn.execute("UPDATE projection_runs SET status='complete' WHERE id=?", (replacement,))
        artifact["certified_bundles"]["6"]["runs"]["xpts_v1"] = replacement

        second = sb.build_scoreboard(conn, artifact=artifact, events=events, top_k=2)
        assert second["identity"]["per_event_runs"]["6"]["xpts_v1"] == replacement
        # Every OTHER event's run is untouched.
        for event in events:
            if event == 6:
                continue
            assert second["identity"]["per_event_runs"][str(event)]["xpts_v1"] == runs["xpts_v1"][event]
        assert sb.scoreboard_digest(first) != sb.scoreboard_digest(second)
    finally:
        conn.close()


def test_the_scoreboard_rejects_a_recorded_run_that_disagrees_with_the_anchor(tmp_path):
    """The agreement check is live: a wrong recorded run must fail, not be ignored."""

    conn = connect_database(tmp_path / "disagree.db")
    try:
        runs, artifact = _multi_event_world(conn)
        assert runs["xpts_v1"][6] != runs["xpts_v1"][5]
        with pytest.raises(sb.ScoreboardError) as failure:
            sb.assert_recorded_runs_agree(
                {"6": {"xpts_v1": runs["xpts_v1"][5]}},     # the WRONG run for event 6
                resolved={"xpts_v1": {6: runs["xpts_v1"][6]}},
            )
        assert "but the anchor resolves" in str(failure.value)

        with pytest.raises(sb.ScoreboardError) as missing:
            sb.assert_recorded_runs_agree({}, resolved={"xpts_v1": {6: runs["xpts_v1"][6]}})
        assert "records no xpts_v1 run for event 6" in str(missing.value)

        # The agreeing case is silent.
        sb.assert_recorded_runs_agree(
            {"6": {"xpts_v1": runs["xpts_v1"][6]}},
            resolved={"xpts_v1": {6: runs["xpts_v1"][6]}},
        )
    finally:
        conn.close()


def test_the_scoreboard_verifies_the_populations_recorded_runs(tmp_path, monkeypatch):
    """The cross-check is WIRED, not merely available.

    A population whose recorded run disagrees with the anchor must make the whole
    scoreboard fail, so a broken recording path cannot quietly produce an artifact
    that names the wrong run.  The tamper is injected at the builder because the two
    real paths cannot disagree -- which is exactly why the check must be proven to
    be called rather than assumed to be reachable.
    """

    import dataclasses

    conn = connect_database(tmp_path / "tamper.db")
    try:
        events = sorted(MULTI_EVENT_FIXTURES)
        runs, artifact = _multi_event_world(conn)
        real_build = wf.build_event_population

        def tampered_build(inner_conn, *, anchor, events, baseline_kind=None, require_same_population=True):
            population = real_build(
                inner_conn, anchor=anchor, events=events, baseline_kind=baseline_kind,
                require_same_population=require_same_population,
            )
            if baseline_kind is None:
                population.identity = dataclasses.replace(
                    population.identity,
                    per_event_runs=tuple(
                        (event, family, 999999 if (event == 6 and family == "xpts_v1") else run)
                        for event, family, run in population.identity.per_event_runs
                    ),
                )
            return population

        monkeypatch.setattr(wf, "build_event_population", tampered_build)
        with pytest.raises(sb.ScoreboardError) as failure:
            sb.build_scoreboard(conn, artifact=artifact, events=events, top_k=2)
        assert "but the anchor resolves" in str(failure.value)
        assert str(runs["xpts_v1"][6]) in str(failure.value)
    finally:
        conn.close()


def test_the_merged_run_map_is_complete_and_event_first(tmp_path):
    conn = connect_database(tmp_path / "merged.db")
    try:
        runs, artifact = _multi_event_world(conn)
        scoreboard = sb.build_scoreboard(conn, artifact=artifact, events=[5, 6], top_k=2)
        merged = scoreboard["identity"]["per_event_runs"]
        assert set(merged) == {"5", "6"}
        for event in ("5", "6"):
            assert merged[event]["xpts_v1"] == runs["xpts_v1"][int(event)]
            assert merged[event][wf.BASELINE_FAMILY] == runs["baseline"][int(event)]
        # The model population's own identity is embedded and agrees.
        embedded = scoreboard["identity"]["walk_forward_identity"]["per_event_runs"]
        assert embedded == {
            "5": {"xpts_v1": runs["xpts_v1"][5]},
            "6": {"xpts_v1": runs["xpts_v1"][6]},
        }
    finally:
        conn.close()
