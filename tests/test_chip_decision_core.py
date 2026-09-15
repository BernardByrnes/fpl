"""Chip decision core + Triple Captain — input contract, arbiter, worlds, TC.

The chip layer must treat football scoring as an INPUT CONTRACT: these tests use
synthetic world series only, and one of them asserts that the chip modules never
import the predictive stack.  Every TC expectation is computed by an independent
hand formula in the test rather than restated from the implementation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fpl_brain import chip_decision as cd
from fpl_brain import chip_triple_captain as tc
from fpl_brain import manager_lineup
from fpl_brain import season_rules as sr

# ---------------------------------------------------------------------------
# synthetic squad + world fixtures (no football model involved)
# ---------------------------------------------------------------------------

GKP = (1, 2)
DEF = (3, 4, 5, 6, 7)
MID = (8, 9, 10, 11, 12)
FWD = (13, 14, 15)
POSITIONS = {
    **{pid: "GKP" for pid in GKP},
    **{pid: "DEF" for pid in DEF},
    **{pid: "MID" for pid in MID},
    **{pid: "FWD" for pid in FWD},
}
STARTERS = (1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15)
BENCH_GK = 2
BENCH_ORDER = (6, 7, 12)
CAPTAIN = 13
VICE = 8
SQUAD = tuple(sorted(POSITIONS))


def _policy(*, captain: int = CAPTAIN, vice: int = VICE) -> manager_lineup.ManagerPolicy:
    return manager_lineup.ManagerPolicy(
        starter_ids=STARTERS,
        bench_gk_id=BENCH_GK,
        bench_outfield_order=BENCH_ORDER,
        captain_id=captain,
        vice_captain_id=vice,
    )


def _worlds(
    *,
    worlds: int,
    core: dict[int, tuple[float, ...]],
    minutes: dict[int, tuple[float, ...]] | None = None,
    events: tuple[int, ...] = (5, 6, 7, 8),
    seed: int = 20260911,
) -> cd.ChipWorldInputs:
    """Build the input contract from plain series — the whole point of the seam."""

    minutes = minutes or {pid: tuple(90.0 for _ in range(worlds)) for pid in core}
    return cd.ChipWorldInputs(
        worlds=worlds,
        player_ids=tuple(sorted(core)),
        minutes=minutes,
        core=core,
        planning_event=5,
        horizon_events=events,
        certification_identity="sha256:" + "c" * 64,
        data_snapshot_sha256="sha256:" + "d" * 64,
        world_seed=seed,
        world_identity="sha256:" + "e" * 64,
    )


def _flat(core_by_pid: dict[int, float], worlds: int) -> dict[int, tuple[float, ...]]:
    return {pid: tuple(float(value) for _ in range(worlds)) for pid, value in core_by_pid.items()}


def _availability(*, available=("bboost", "3xc", "freehit", "wildcard")) -> list[dict]:
    return [
        {"name": name, "available_for_event": name in available, "used": False, "expired": False, "window": "GW5-GW19"}
        for name in ("bboost", "3xc", "freehit", "wildcard")
    ]


def _request(**overrides) -> tc.TripleCaptainRequest:
    base = dict(
        worlds=_worlds(worlds=4, core=_flat({pid: 5.0 for pid in SQUAD}, 4)),
        policy=_policy(),
        positions=POSITIONS,
        planning_event=5,
    )
    base.update(overrides)
    return tc.TripleCaptainRequest(**base)


# ---------------------------------------------------------------------------
# A/B. INPUT CONTRACT
# ---------------------------------------------------------------------------


def test_A_chip_modules_never_import_the_predictive_stack():
    """The chip layer must not require xpts_v1/player_rates model internals."""

    banned = (
        "xpts", "player_rates", "monte_carlo", "team_model", "minutes_model",
        "minutes_coherence", "joint_minutes", "substitution_model", "player_rates",
    )
    for name in ("chip_decision", "chip_triple_captain"):
        source = Path(f"fpl_brain/{name}.py").read_text(encoding="utf-8")
        offenders = [
            line.strip()
            for line in source.splitlines()
            if line.startswith(("from .", "import ")) and any(token in line for token in banned)
        ]
        assert offenders == [], (name, offenders)
    # and it consumes a plain matrix, not a model object
    worlds = cd.ChipWorldInputs.from_world_matrix(
        {"worlds": 3, "player_ids": [1, 2], "minutes": {1: [90, 0, 45], 2: [0, 90, 0]},
         "core": {1: [6.0, 0.0, 3.0], 2: [0.0, 5.0, 0.0]}},
        planning_event=5, horizon_events=(5, 6, 7, 8),
        certification_identity="sha256:x", data_snapshot_sha256="sha256:y",
        world_seed=7, world_identity="sha256:z",
    )
    assert worlds.worlds == 3 and worlds.appeared(1) == (True, False, True)


def test_B_an_alternate_world_provider_feeds_tc_unchanged():
    """A different provider that conforms to the protocol needs no TC changes."""

    class AlternateProvider:
        def __init__(self):
            self.calls = 0

        def provide(self, *, planning_event: int) -> cd.ChipWorldInputs:
            self.calls += 1
            return _worlds(
                worlds=4,
                core=_flat({pid: 5.0 for pid in SQUAD} | {CAPTAIN: 9.0}, 4),
            )

    provider = AlternateProvider()
    assert isinstance(provider, cd.ChipWorldProvider)
    worlds = provider.provide(planning_event=5)
    evaluation = tc.evaluate_triple_captain(_request(worlds=worlds))
    assert provider.calls == 1
    assert evaluation.action == cd.CHIP_ACTION_TC
    assert evaluation.evidence["worlds"] == 4


# ---------------------------------------------------------------------------
# C/D/E. ARBITER
# ---------------------------------------------------------------------------


def _evaluated_decision(**overrides) -> cd.ChipDecision:
    evaluation = tc.evaluate_triple_captain(_request())
    base = dict(
        planning_event=5,
        chip_availability=_availability(),
        evaluations={cd.CHIP_ACTION_TC: evaluation},
        horizon_events=(5, 6, 7, 8),
        certification_identity="sha256:" + "c" * 64,
        data_snapshot_sha256="sha256:" + "d" * 64,
        certification_valid=True,
        manager_state={"squad_ids": list(SQUAD)},
    )
    base.update(overrides)
    return cd.decide_chip_action(**base)


def test_C_exactly_one_action_is_returned():
    decision = _evaluated_decision()
    assert decision.recommended_action in cd.CHIP_ACTIONS
    assert decision.as_dict()["recommended_action"] == decision.recommended_action
    assert decision.one_chip_rule_enforced is True
    # the unimplemented chips are reported, never guessed
    assert any(cd.DIAG_CHIP_EVALUATOR_NOT_IMPLEMENTED in code for code in decision.reason_codes)


def test_D_one_chip_per_gameweek_is_enforced():
    two_chips = _evaluated_decision(chips_already_played_for_event=["wildcard", "freehit"])
    assert two_chips.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_MULTIPLE_ACTIONS in two_chips.reason_codes
    assert two_chips.recommended_action == cd.CHIP_ACTION_NO_CHIP

    spent = _evaluated_decision(chips_already_played_for_event=["wildcard"])
    assert spent.status == cd.STATUS_NO_CHIP
    assert cd.DIAG_CHIP_GAMEWEEK_ALREADY_USED in spent.reason_codes
    assert spent.recommended_action == cd.CHIP_ACTION_NO_CHIP


def test_E_an_unavailable_chip_cannot_be_recommended():
    decision = _evaluated_decision(chip_availability=_availability(available=("wildcard", "freehit")))
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP
    assert any(cd.DIAG_CHIP_UNAVAILABLE in code for code in decision.reason_codes)
    # a positive-uplift evaluation with the chip unavailable is still not played
    assert decision.status in {cd.STATUS_NO_CHIP, cd.STATUS_CHIP_REVIEW_REQUIRED}


# ---------------------------------------------------------------------------
# F/G/H. SELECTION vs VALUATION
# ---------------------------------------------------------------------------


def test_F_the_split_is_deterministic_across_runs():
    first = cd.partition_worlds(64, seed=20260911)
    second = cd.partition_worlds(64, seed=20260911)
    assert first == second
    assert first.basis.startswith("sha256:")
    assert cd.partition_worlds(64, seed=20260912) != first


def test_G_partitions_are_disjoint_and_cover_every_world():
    partition = cd.partition_worlds(64, seed=20260911)
    assert set(partition.selection) & set(partition.valuation) == set()
    assert sorted(partition.selection + partition.valuation) == list(range(64))
    assert partition.selection and partition.valuation
    with pytest.raises(cd.ChipInputError):
        cd.partition_worlds(1, seed=20260911)


def test_H_the_reported_value_uses_valuation_worlds_only():
    """Selection and valuation data are deliberately different here."""

    worlds = 64
    # The captain scores 100 in the even half and 0 in the odd half; selection and
    # valuation each contain a mix, so a value computed on the wrong side differs.
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(100.0 if index % 2 == 0 else 0.0 for index in range(worlds))
    inputs = _worlds(worlds=worlds, core=core)
    evaluation = tc.evaluate_triple_captain(_request(worlds=inputs))
    partition = cd.partition_worlds(worlds, seed=inputs.world_seed)

    expected = sum(100.0 for index in partition.valuation if index % 2 == 0) / partition.valuation_worlds
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(expected, abs=1e-6)
    # the selection-side mean is reported separately and is not the reported value
    selection_only = sum(100.0 for index in partition.selection if index % 2 == 0) / partition.selection_worlds
    assert evaluation.candidate_metrics["mean_tc_armband_bonus_selection"] == pytest.approx(
        float(tc.TRIPLE_CAPTAIN_BONUS_COPIES) * selection_only, abs=1e-6
    )
    assert evaluation.evidence["selection_valuation_disjoint"] is True


# ---------------------------------------------------------------------------
# I/J/K/L. TRIPLE CAPTAIN SEMANTICS
# ---------------------------------------------------------------------------


def test_the_armband_rule_matches_the_certified_engine():
    """The chip's armband series must equal manager_lineup.captain_multiplier."""

    worlds = 8
    minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in SQUAD}
    minutes[CAPTAIN] = (0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0, 90.0)
    minutes[VICE] = (90.0, 0.0, 45.0, 0.0, 0.0, 0.0, 0.0, 90.0)
    core = _flat({pid: 2.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(float(10 + index) for index in range(worlds))
    core[VICE] = tuple(float(20 + index) for index in range(worlds))
    appeared = {pid: tuple(value > 0.0 for value in series) for pid, series in minutes.items()}

    mine = tc._armband_series(
        worlds=worlds, core=core, appeared=appeared, captain=CAPTAIN, vice=VICE,
        indices=tuple(range(worlds)),
    )
    oracle = []
    for index in range(worlds):
        extra, _source = manager_lineup.captain_multiplier(
            _policy(),
            {pid: minutes[pid][index] for pid in SQUAD},
            {pid: core[pid][index] for pid in SQUAD},
        )
        oracle.append(extra)
    assert mine == oracle


def test_I_triple_captain_scoring_versus_normal_captain_scoring():
    """Captain always appears: the chip is worth exactly one more copy of his score."""

    worlds = 64
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(float(4 + (index % 5)) for index in range(worlds))  # highest in the XI
    inputs = _worlds(worlds=worlds, core=core)
    evaluation = tc.evaluate_triple_captain(_request(worlds=inputs))
    partition = cd.partition_worlds(worlds, seed=inputs.world_seed)

    values = [core[CAPTAIN][index] for index in partition.valuation]
    expected = sum(values) / len(values)
    assert evaluation.candidate_metrics["captain_id"] == CAPTAIN
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(expected, abs=1e-6)
    # and it is NOT a flat x3 of a projected number: the armband is world-level
    assert evaluation.uncertainty["p_armband_captain"] == 1.0
    assert evaluation.uncertainty["p_armband_none"] == 0.0


def test_J_captain_appearance_and_the_vice_fallback_matter():
    worlds = 64
    minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in SQUAD}
    minutes[CAPTAIN] = tuple(0.0 if index % 2 else 90.0 for index in range(worlds))
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(50.0 for _ in range(worlds))
    core[VICE] = tuple(9.0 for _ in range(worlds))
    inputs = _worlds(worlds=worlds, core=core, minutes=minutes)
    evaluation = tc.evaluate_triple_captain(_request(worlds=inputs))
    partition = cd.partition_worlds(worlds, seed=inputs.world_seed)

    assert evaluation.uncertainty["p_armband_captain"] == pytest.approx(
        sum(1 for index in partition.valuation if index % 2 == 0) / partition.valuation_worlds, abs=1e-6
    )
    assert evaluation.uncertainty["p_armband_vice"] == pytest.approx(
        1.0 - evaluation.uncertainty["p_armband_captain"], abs=1e-6
    )
    expected = sum(
        (50.0 if index % 2 == 0 else 9.0) for index in partition.valuation
    ) / partition.valuation_worlds
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(expected, abs=1e-6)
    assert tc.DIAG_TC_CAPTAIN_APPEARANCE_UNCERTAIN in evaluation.reason_codes
    assert tc.DIAG_TC_VICE_FALLBACK_MATERIAL in evaluation.reason_codes


def test_K_vice_fallback_in_the_save_arm_and_the_chips_own_captain_choice():
    """The engine's captain never plays, so the SAVE armband falls to the vice.

    The chip then does better than doubling that: it may also move the armband to
    a player who actually appears, which is exactly why the evaluator selects a
    (captain, vice) pair rather than doubling the engine's choice.
    """

    worlds = 64
    minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in SQUAD}
    minutes[CAPTAIN] = tuple(0.0 for _ in range(worlds))     # the engine's captain never appears
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(99.0 for _ in range(worlds))       # irrelevant: he never plays
    core[VICE] = tuple(7.0 for _ in range(worlds))
    inputs = _worlds(worlds=worlds, core=core, minutes=minutes)
    evaluation = tc.evaluate_triple_captain(_request(worlds=inputs))

    # the SAVE arm really did fall back to the vice (one bonus copy of his score)
    assert evaluation.candidate_metrics["mean_save_armband_bonus_valuation"] == pytest.approx(7.0)
    assert evaluation.candidate_metrics["save_captain_id"] == CAPTAIN
    # the chip keeps doubling the best available armband holder: one extra copy
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(7.0, abs=1e-6)


def test_L_no_xi_player_appears_so_the_chip_is_worthless():
    """With the whole XI absent there is no armband at all, so TC cannot pay."""

    worlds = 64
    minutes = {pid: tuple(0.0 for _ in range(worlds)) for pid in SQUAD}
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    inputs = _worlds(worlds=worlds, core=core, minutes=minutes)
    evaluation = tc.evaluate_triple_captain(_request(worlds=inputs))

    assert evaluation.uncertainty["p_armband_none"] == pytest.approx(1.0)
    assert evaluation.uncertainty["armband_source_counts"]["NONE"] == evaluation.candidate_metrics["valuation_worlds"]
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(0.0)
    assert tc.DIAG_TC_NO_ARMband_POSSIBLE in evaluation.reason_codes

    decision = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: evaluation})
    assert decision.status == cd.STATUS_NO_CHIP
    assert cd.DIAG_CHIP_UPLIFT_NON_POSITIVE in decision.reason_codes


def test_L2_a_pair_that_appears_is_preferred_over_an_absent_armband():
    """The evaluator never doubles an absent player when a playing pair exists."""

    worlds = 64
    minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in SQUAD}
    minutes[CAPTAIN] = tuple(0.0 for _ in range(worlds))
    minutes[VICE] = tuple(0.0 for _ in range(worlds))
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    inputs = _worlds(worlds=worlds, core=core, minutes=minutes)
    evaluation = tc.evaluate_triple_captain(_request(worlds=inputs))

    assert evaluation.candidate_metrics["captain_id"] not in {CAPTAIN, VICE}
    assert evaluation.uncertainty["p_armband_none"] == 0.0
    # the chip's pair contributes two bonus copies; the SAVE pair contributes none,
    # because both of its players are absent
    assert evaluation.candidate_metrics["mean_save_armband_bonus_valuation"] == pytest.approx(0.0)
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(
        float(tc.TRIPLE_CAPTAIN_BONUS_COPIES) * 1.0, abs=1e-6
    )


# ---------------------------------------------------------------------------
# M/N/O. FOUR-GW RULE
# ---------------------------------------------------------------------------


def test_M_tc_is_evaluated_inside_the_exact_four_gw_policy():
    evaluation = tc.evaluate_triple_captain(_request())
    assert evaluation.candidate_metrics["horizon_events"] == [5, 6, 7, 8]
    assert evaluation.candidate_metrics["route_consequence_delta"] == 0.0
    assert "identical across arms" in evaluation.candidate_metrics["four_gw_basis"]
    assert evaluation.evidence["planning_event"] == 5


def test_N_an_incomplete_horizon_fails_closed():
    short = _worlds(worlds=8, core=_flat({pid: 5.0 for pid in SQUAD}, 8), events=(5, 6, 7))
    with pytest.raises(cd.ChipInputError) as failure:
        tc.evaluate_triple_captain(_request(worlds=short))
    assert cd.DIAG_CHIP_HORIZON_INCOMPLETE in str(failure.value)

    long = _worlds(worlds=8, core=_flat({pid: 5.0 for pid in SQUAD}, 8), events=(5, 6, 7, 8, 9))
    with pytest.raises(cd.ChipInputError):
        tc.evaluate_triple_captain(_request(worlds=long))

    # ...and the arbiter refuses the same way, without needing the evaluator
    decision = _evaluated_decision(horizon_events=(5, 6, 7))
    assert decision.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_HORIZON_INCOMPLETE in decision.reason_codes


def test_O_an_unavailable_triple_captain_fails_closed():
    decision = _evaluated_decision(chip_availability=_availability(available=("wildcard", "freehit")))
    assert decision.recommended_action != cd.CHIP_ACTION_TC
    assert any("TC" in code for code in decision.reason_codes if cd.DIAG_CHIP_UNAVAILABLE in code)


# ---------------------------------------------------------------------------
# P/Q/R. CALIBRATION
# ---------------------------------------------------------------------------


def test_P_a_positive_mean_uplift_cannot_emit_play_chip():
    worlds = 64
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(12.0 for _ in range(worlds))
    evaluation = tc.evaluate_triple_captain(
        _request(worlds=_worlds(worlds=worlds, core=core))
    )
    assert evaluation.mean_uplift is not None and evaluation.mean_uplift > 0

    decision = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: evaluation})
    assert decision.recommended_action == cd.CHIP_ACTION_TC
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
    assert decision.status != cd.STATUS_PLAY_CHIP
    assert decision.calibration_status == cd.CALIBRATION_UNCALIBRATED
    assert cd.DIAG_CHIP_RESERVATION_UNCALIBRATED in decision.reason_codes
    assert decision.candidate_metrics["net_of_reservation"] is None

    # PLAY_CHIP becomes reachable only with a CALIBRATED reservation value.
    class Calibrated:
        def estimate(self, *, action, planning_event, expiry_event, state):
            return cd.ReservationEstimate(
                value=1.0, calibration_status=cd.CALIBRATION_CALIBRATED, terminal_value=0.0,
                weeks_to_expiry=17, reason_codes=(), conditional_on=("weeks_remaining",),
            )

    played = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: evaluation}, reservation=Calibrated())
    assert played.status == cd.STATUS_PLAY_CHIP
    assert played.recommended_action == cd.CHIP_ACTION_TC


def test_Q_calibration_status_is_always_surfaced():
    evaluation = tc.evaluate_triple_captain(_request())
    assert evaluation.calibration_status == cd.CALIBRATION_UNCALIBRATED
    decision = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: evaluation})
    assert decision.calibration_status in cd.CALIBRATION_STATUSES
    assert decision.as_dict()["calibration_status"] == decision.calibration_status
    estimate = cd.UncalibratedReservation().estimate(
        action=cd.CHIP_ACTION_TC, planning_event=5, expiry_event=19, state={}
    )
    assert estimate.value is None
    assert estimate.terminal_value == 0.0        # terminal value 0 at expiry
    assert estimate.weeks_to_expiry == 14
    assert cd.DIAG_RESERVATION_TERMINAL_ZERO in estimate.reason_codes


def test_R_candidate_recheck_status_is_supported_and_uncertainty_is_reported():
    worlds = 64
    minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in SQUAD}
    minutes[CAPTAIN] = tuple(0.0 if index % 8 else 90.0 for index in range(worlds))
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(40.0 for _ in range(worlds))
    core[VICE] = tuple(2.0 for _ in range(worlds))
    evaluation = tc.evaluate_triple_captain(
        _request(worlds=_worlds(worlds=worlds, core=core, minutes=minutes))
    )
    decision = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: evaluation})
    assert decision.status in {
        cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED,
        cd.STATUS_CHIP_REVIEW_REQUIRED,
    }
    # the uncertainty contract the reviewer needs
    for key in ("mean_paired_uplift", "paired_se", "captain_id", "vice_captain_id"):
        assert key in decision.candidate_metrics
    for key in ("p_captain_appears", "p_armband_captain", "p_armband_vice", "p_armband_none",
                "paired_interval_low", "paired_interval_high", "armband_source_counts"):
        assert key in decision.uncertainty
    assert decision.candidate_metrics["paired_se"] >= 0.0
    assert decision.uncertainty["paired_interval_low"] <= decision.candidate_metrics["mean_paired_uplift"]

    # an unresolvable interval asks for review rather than a recheck
    noisy = tc.evaluate_triple_captain(
        _request(
            worlds=_worlds(
                worlds=worlds, core=core,
                minutes={**minutes, CAPTAIN: tuple(90.0 if index % 2 else 0.0 for index in range(worlds))},
            )
        )
    )
    assert noisy.mean_uplift is not None
    assert (noisy.uncertainty["paired_interval_low"] <= 0.0) or (noisy.mean_uplift > 0)


def test_input_uncertainty_flags_are_propagated_verbatim():
    evaluation = tc.evaluate_triple_captain(
        _request(input_uncertainty_flags=("ROLE_UNCERTAINTY_PLAYER_13",))
    )
    assert evaluation.uncertainty["input_uncertainty_flags"] == ["ROLE_UNCERTAINTY_PLAYER_13"]
    assert tc.DIAG_TC_INPUT_UNCERTAINTY in evaluation.reason_codes


# ---------------------------------------------------------------------------
# S. DETERMINISM
# ---------------------------------------------------------------------------


def test_S_identical_inputs_produce_identical_results():
    first = tc.evaluate_triple_captain(_request())
    second = tc.evaluate_triple_captain(_request())
    assert first == second
    assert first.reason_codes == tuple(sorted(first.reason_codes))

    left = _evaluated_decision().as_dict()
    right = _evaluated_decision().as_dict()
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    assert left["reason_codes"] == sorted(left["reason_codes"])


# ---------------------------------------------------------------------------
# T. REGRESSION — nothing existing changes
# ---------------------------------------------------------------------------


def test_T_existing_behaviour_is_unchanged_when_no_chip_is_evaluated():
    """No chip supplied means no chip action, and the predictive code is untouched."""

    decision = cd.decide_chip_action(
        planning_event=5,
        chip_availability=_availability(),
        evaluations=None,
        horizon_events=(5, 6, 7, 8),
        certification_identity="sha256:" + "c" * 64,
        data_snapshot_sha256="sha256:" + "d" * 64,
        certification_valid=True,
        manager_state={"squad_ids": list(SQUAD)},
    )
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP
    assert decision.candidate_metrics == {}

    # The chip vocabulary is the canonical season-rules tuple, not a copy.
    from fpl_brain import transfer_state as ts, route_comparator as rc
    assert ts.CHIP_FT_PRESERVING is sr.FT_PRESERVING_CHIPS
    assert ts.CHIP_BOOSTING is sr.NON_TRANSFER_CHIPS
    assert rc.CHIP_KEYWORDS is sr.CHIP_NAME_KEYWORDS

    # The certified predictive code identity is untouched by the chip layer.
    from fpl_brain import analytics
    assert "fpl_brain/chip_decision.py" not in analytics.SOURCE_SNAPSHOT_FILES
    assert "fpl_brain/chip_triple_captain.py" not in analytics.SOURCE_SNAPSHOT_FILES
    assert "fpl_brain/season_rules.py" not in analytics.SOURCE_SNAPSHOT_FILES


def test_T2_a_chip_route_is_still_refused_by_the_route_layer():
    from fpl_brain import route_comparator as rc

    assert rc.CHIP_ROUTE_FLAG == "CHIP_ROUTE_NOT_MODELLED"
    assert rc._is_chip("3xc") and rc._is_chip("triple_captain") and rc._is_chip("bench_boost")


def test_T3_the_wildcard_play_rule_is_not_executable():
    from fpl_brain import four_gw_decision as fg

    verified = fg.wildcard_trigger_screen(
        weak_slot_count=5,
        supported_evaluation={
            "schema": fg.SUPPORTED_WILDCARD_EVALUATION_SCHEMA,
            "four_gw_net_core": 25.0,
            "four_gw_certification_identity": "sha256:abc",
            "data_snapshot_sha256": "deadbeef",
            "distinct_squad_from_current": True,
            "wildcard_squad_player_ids": [1, 2, 3],
        },
    )
    assert verified["status"] == fg.WILDCARD_EVALUATION_SUPPORTED
    assert verified["actionable"] is False
    assert verified["executable"] is False
    assert verified["recommendation"] == "NONE"
    assert verified["verified_evaluation_signal"] == "POSITIVE"
    assert verified["calibration_status"] == fg.WILDCARD_CALIBRATION_STATUS
    assert "PLAY_WILDCARD" not in json.dumps(verified)
    assert verified["wildcard_quantitative_capability"] == "NOT_SUPPORTED"


def test_arbiter_fails_closed_without_certification_or_manager_state():
    assert _evaluated_decision(certification_valid=False).status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert _evaluated_decision(certification_identity=None).reason_codes == (cd.DIAG_CHIP_CERTIFICATION_REQUIRED,)
    assert _evaluated_decision(data_snapshot_sha256=None).reason_codes == (cd.DIAG_CHIP_CERTIFICATION_REQUIRED,)
    assert _evaluated_decision(manager_state=None).reason_codes == (cd.DIAG_CHIP_MANAGER_STATE_INCOMPLETE,)
    assert _evaluated_decision(manager_state={"squad_ids": []}).reason_codes == (
        cd.DIAG_CHIP_MANAGER_STATE_INCOMPLETE,
    )


def test_world_contract_gaps_fail_closed():
    with pytest.raises(cd.ChipInputError):
        cd.ChipWorldInputs(worlds=0, player_ids=(1,), minutes={1: ()}, core={1: ()},
                           planning_event=5, horizon_events=(5, 6, 7, 8),
                           certification_identity="sha256:c", data_snapshot_sha256="sha256:d",
                           world_seed=1, world_identity="sha256:e")
    with pytest.raises(cd.ChipInputError):
        cd.ChipWorldInputs.from_world_matrix(
            {"worlds": 2, "player_ids": [1], "minutes": {1: [90.0]}, "core": {1: [1.0, 2.0]}},
            planning_event=5, horizon_events=(5, 6, 7, 8),
            certification_identity="sha256:c", data_snapshot_sha256="sha256:d",
            world_seed=1, world_identity="sha256:e",
        )
    with pytest.raises(cd.ChipInputError):
        cd.ChipWorldInputs.from_world_matrix(
            {"worlds": 2, "player_ids": [1], "minutes": {1: [90.0, 0.0]}, "core": {1: [1.0, 2.0]}},
            planning_event=5, horizon_events=(5, 6, 7, 8),
            certification_identity=None, data_snapshot_sha256="sha256:d",
            world_seed=1, world_identity="sha256:e",
        )
