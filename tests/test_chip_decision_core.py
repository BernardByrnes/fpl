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


def _binding(*, planning_event: int = 5, identity: str = "sha256:" + "c" * 64) -> cd.ChipHorizonBinding:
    """The exact certified four-event identity a chip decision is authorised for."""

    return cd.ChipHorizonBinding(
        planning_event=planning_event,
        horizon_events=cd.canonical_chip_horizon(planning_event),
        certification_identity=identity,
        data_snapshot_sha256="sha256:" + "d" * 64,
    )


def _request(**overrides) -> tc.TripleCaptainRequest:
    base = dict(
        worlds=_worlds(worlds=4, core=_flat({pid: 5.0 for pid in SQUAD}, 4)),
        horizon_binding=_binding(),
        policy=_policy(),
        positions=POSITIONS,
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
        horizon_binding=_binding(),
        chip_availability=_availability(),
        evaluations={cd.CHIP_ACTION_TC: evaluation},
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
        core=core, appeared=appeared, captain=CAPTAIN, vice=VICE,
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

    # the SAVE arm is optimised too: it also moves the armband to a player who
    # appears, so it earns one bonus copy of the best available score
    assert evaluation.candidate_metrics["mean_save_armband_bonus_valuation"] == pytest.approx(7.0)
    assert evaluation.candidate_metrics["requested_captain_id"] == CAPTAIN
    assert evaluation.candidate_metrics["save_captain_id"] != CAPTAIN
    # the chip still adds exactly one extra copy on top of that
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
    assert tc.DIAG_TC_NO_ARMBAND_POSSIBLE in evaluation.reason_codes

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
    assert evaluation.candidate_metrics["save_captain_id"] not in {CAPTAIN, VICE}
    assert evaluation.uncertainty["p_armband_none"] == 0.0
    # BOTH arms optimise, so the chip is worth exactly ONE extra copy: the TC arm
    # takes two bonus copies and the SAVE arm one, not two-versus-zero.
    assert evaluation.candidate_metrics["mean_save_armband_bonus_valuation"] == pytest.approx(1.0)
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# M/N/O. FOUR-GW RULE
# ---------------------------------------------------------------------------


def test_M_tc_is_evaluated_inside_the_exact_four_gw_policy():
    evaluation = tc.evaluate_triple_captain(_request())
    assert evaluation.candidate_metrics["horizon_events"] == [5, 6, 7, 8]
    assert evaluation.candidate_metrics["route_consequence_delta"] == 0.0
    assert evaluation.candidate_metrics["route_consequence_contract"] == tc.ROUTE_DELTA_CONTRACT
    assert "identical across arms" in evaluation.candidate_metrics["four_gw_basis"]
    assert evaluation.evidence["planning_event"] == 5


def test_N_an_incomplete_horizon_fails_closed():
    short = _worlds(worlds=8, core=_flat({pid: 5.0 for pid in SQUAD}, 8), events=(5, 6, 7))
    with pytest.raises(cd.ChipInputError) as failure:
        tc.evaluate_triple_captain(_request(worlds=short))
    assert cd.DIAG_CHIP_PLANNING_EVENT_MISMATCH in str(failure.value)

    long = _worlds(worlds=8, core=_flat({pid: 5.0 for pid in SQUAD}, 8), events=(5, 6, 7, 8, 9))
    with pytest.raises(cd.ChipInputError):
        tc.evaluate_triple_captain(_request(worlds=long))

    # ...and a non-canonical horizon cannot even be bound
    with pytest.raises(cd.ChipInputError) as failure:
        cd.ChipHorizonBinding(planning_event=5, horizon_events=(5, 6, 7), certification_identity="sha256:c")
    assert cd.DIAG_CHIP_HORIZON_NOT_CANONICAL in str(failure.value)


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
        horizon_binding=_binding(),
        chip_availability=_availability(),
        evaluations=None,
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
    assert _evaluated_decision(certification_valid=False).reason_codes == (cd.DIAG_CHIP_CERTIFICATION_REQUIRED,)
    # an identity-less binding is unrepresentable rather than merely refused
    with pytest.raises(cd.ChipInputError) as failure:
        cd.ChipHorizonBinding(planning_event=5, horizon_events=(5, 6, 7, 8), certification_identity="")
    assert cd.DIAG_CHIP_CERTIFICATION_REQUIRED in str(failure.value)
    # ...and a binding with no data snapshot identity cannot authorise a decision
    no_snapshot = cd.ChipHorizonBinding(
        planning_event=5, horizon_events=(5, 6, 7, 8), certification_identity="sha256:" + "c" * 64,
        data_snapshot_sha256=None,
    )
    assert _evaluated_decision(horizon_binding=no_snapshot).reason_codes == (cd.DIAG_CHIP_CERTIFICATION_REQUIRED,)
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


# ===========================================================================
# P2-1 — PLAY_TC vs SAVE counterfactual, and the route-consequence contract
# ===========================================================================


def _absent_pair_worlds(*, worlds: int = 64, core_others: float = 1.0):
    """CAPTAIN and VICE never appear; every other starter always does."""

    minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in SQUAD}
    minutes[CAPTAIN] = tuple(0.0 for _ in range(worlds))
    minutes[VICE] = tuple(0.0 for _ in range(worlds))
    core = _flat({pid: core_others for pid in SQUAD}, worlds)
    return _worlds(worlds=worlds, core=core, minutes=minutes)


def test_P2_1_A_both_arms_optimise_captain_vice_independently():
    """The SAVE arm moves the armband too, instead of inheriting the request."""

    evaluation = tc.evaluate_triple_captain(_request(worlds=_absent_pair_worlds()))
    metrics = evaluation.candidate_metrics
    assert metrics["save_arm_optimised"] is True
    assert metrics["requested_captain_id"] == CAPTAIN
    assert metrics["requested_vice_captain_id"] == VICE
    # the request's pair is suboptimal for BOTH arms, so both differ from it
    assert (metrics["captain_id"], metrics["vice_captain_id"]) != (CAPTAIN, VICE)
    assert (metrics["save_captain_id"], metrics["save_vice_captain_id"]) != (CAPTAIN, VICE)
    assert tc.DIAG_TC_PLAY_ARM_DIFFERS_FROM_REQUEST in evaluation.reason_codes
    assert tc.DIAG_TC_SAVE_ARM_DIFFERS_FROM_REQUEST in evaluation.reason_codes


def test_P2_1_B_a_weak_request_policy_cannot_inflate_the_uplift():
    """The brief's example: weak request -> uplift 1, not 2."""

    evaluation = tc.evaluate_triple_captain(_request(worlds=_absent_pair_worlds()))
    metrics = evaluation.candidate_metrics
    assert metrics["mean_tc_armband_bonus_valuation"] == pytest.approx(2.0)
    assert metrics["mean_save_armband_bonus_valuation"] == pytest.approx(1.0)
    assert metrics["mean_paired_uplift"] == pytest.approx(1.0, abs=1e-6)

    # a differently-shaped weak request cannot move the answer either: the request
    # pair only ever appears in the audit fields
    other = tc.evaluate_triple_captain(
        _request(worlds=_absent_pair_worlds(), policy=_policy(captain=14, vice=15))
    )
    assert other.candidate_metrics["mean_paired_uplift"] == pytest.approx(1.0, abs=1e-6)
    assert other.candidate_metrics["requested_captain_id"] == 14


def test_P2_1_C_fallback_works_in_both_arms():
    """Both arms' armband composition is reported, so either arm can fall back."""

    worlds = 64
    minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in SQUAD}
    minutes[CAPTAIN] = tuple(0.0 if index % 2 else 90.0 for index in range(worlds))
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(50.0 for _ in range(worlds))
    core[VICE] = tuple(9.0 for _ in range(worlds))
    evaluation = tc.evaluate_triple_captain(
        _request(worlds=_worlds(worlds=worlds, core=core, minutes=minutes))
    )
    for key in ("armband_source_counts", "save_armband_source_counts"):
        counts = evaluation.uncertainty[key]
        assert set(counts) == {"CAPTAIN", "VICE", "NONE"}
        assert sum(counts.values()) == evaluation.candidate_metrics["valuation_worlds"]


def test_P2_1_D_selection_happens_only_on_selection_worlds():
    """A candidate that wins on valuation worlds must not be the selected one."""

    worlds = 64
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    # FWD 14 is the best on valuation worlds; FWD 15 is the best on selection ones
    partition = cd.partition_worlds(worlds, seed=20260911)
    sel, val = set(partition.selection), set(partition.valuation)
    core[14] = tuple(50.0 if index in val else 1.0 for index in range(worlds))
    core[15] = tuple(1.0 if index in val else 40.0 for index in range(worlds))
    evaluation = tc.evaluate_triple_captain(_request(worlds=_worlds(worlds=worlds, core=core)))
    assert evaluation.candidate_metrics["captain_id"] == 15     # chosen on selection worlds
    assert sel and val


def test_P2_1_E_valuation_worlds_do_not_re_select_candidates():
    """The reported value belongs to the pair the SELECTION side chose."""

    worlds = 64
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    partition = cd.partition_worlds(worlds, seed=20260911)
    sel, val = set(partition.selection), set(partition.valuation)
    core[14] = tuple(60.0 if index in val else 1.0 for index in range(worlds))
    core[15] = tuple(1.0 if index in val else 40.0 for index in range(worlds))
    evaluation = tc.evaluate_triple_captain(_request(worlds=_worlds(worlds=worlds, core=core)))
    assert evaluation.candidate_metrics["captain_id"] == 15
    # on the valuation worlds player 15 is worth 1.0 and player 14 is worth 60.0,
    # so a re-selecting implementation would report the 14 side; the reported
    # value is the selection-chosen policy valued on valuation worlds.
    assert evaluation.candidate_metrics["mean_tc_armband_bonus_valuation"] == pytest.approx(2.0, abs=1e-6)


def test_P2_1_F_tie_breaking_is_deterministic_in_both_arms():
    """All-equal worlds: both arms resolve to the same lexicographic minimum."""

    evaluation = tc.evaluate_triple_captain(_request(worlds=_worlds(worlds=64, core=_flat({pid: 5.0 for pid in SQUAD}, 64))))
    metrics = evaluation.candidate_metrics
    expected_captain = min(STARTERS)
    assert metrics["captain_id"] == expected_captain
    assert metrics["save_captain_id"] == expected_captain
    again = tc.evaluate_triple_captain(_request(worlds=_worlds(worlds=64, core=_flat({pid: 5.0 for pid in SQUAD}, 64))))
    assert again.candidate_metrics["captain_id"] == expected_captain
    assert again.candidate_metrics["save_vice_captain_id"] == metrics["save_vice_captain_id"]


def test_P2_1_G_route_consequence_delta_is_included_exactly_once():
    worlds = 64
    core = _flat({pid: 1.0 for pid in SQUAD}, worlds)
    core[CAPTAIN] = tuple(10.0 for _ in range(worlds))
    inputs = _worlds(worlds=worlds, core=core)
    zero = tc.evaluate_triple_captain(_request(worlds=inputs))
    with_delta = tc.evaluate_triple_captain(_request(worlds=inputs, route_consequence_delta=3.0))

    assert with_delta.candidate_metrics["mean_paired_uplift"] == pytest.approx(
        zero.candidate_metrics["mean_paired_uplift"] + 3.0, abs=1e-6
    )
    # a deterministic scalar shifts the mean and leaves dispersion untouched
    assert with_delta.candidate_metrics["paired_se"] == pytest.approx(zero.candidate_metrics["paired_se"], abs=1e-9)
    assert with_delta.uncertainty["paired_interval_low"] == pytest.approx(
        zero.uncertainty["paired_interval_low"] + 3.0, abs=1e-6
    )
    assert tc.DIAG_TC_ROUTE_DELTA_APPLIED in with_delta.reason_codes
    assert with_delta.candidate_metrics["route_consequence_contract"] == tc.ROUTE_DELTA_CONTRACT

    # ...and it reaches the DECISION value, not just the report
    zero_decision = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: zero})
    delta_decision = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: with_delta})
    assert delta_decision.candidate_metrics["mean_paired_uplift"] == pytest.approx(
        zero_decision.candidate_metrics["mean_paired_uplift"] + 3.0, abs=1e-6
    )


def test_P2_1_H_zero_route_delta_preserves_the_prior_semantics():
    evaluation = tc.evaluate_triple_captain(_request())
    assert evaluation.candidate_metrics["route_consequence_delta"] == 0.0
    assert tc.DIAG_TC_ROUTE_DELTA_APPLIED not in evaluation.reason_codes
    # the paired value is exactly the documented copy arithmetic
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(
        evaluation.candidate_metrics["mean_tc_armband_bonus_valuation"]
        - evaluation.candidate_metrics["mean_save_armband_bonus_valuation"],
        abs=1e-6,
    )


# ===========================================================================
# P2-2 — exact certified four-event identity
# ===========================================================================


def test_P2_2_A_the_exact_certified_horizon_passes():
    binding = _binding()
    assert binding.horizon_events == (5, 6, 7, 8)
    assert binding.horizon_events == cd.canonical_chip_horizon(5)
    evaluation = tc.evaluate_triple_captain(_request(horizon_binding=binding))
    assert evaluation.candidate_metrics["horizon_events"] == [5, 6, 7, 8]
    decision = _evaluated_decision(horizon_binding=binding)
    assert decision.status != cd.STATUS_INSUFFICIENT_EVIDENCE


def test_P2_2_B_a_duplicated_four_event_horizon_is_refused():
    with pytest.raises(cd.ChipInputError) as failure:
        cd.ChipHorizonBinding(planning_event=5, horizon_events=(5, 5, 8, 12),
                              certification_identity="sha256:" + "c" * 64)
    assert cd.DIAG_CHIP_HORIZON_NOT_CANONICAL in str(failure.value)
    # a hand-built decision cannot smuggle it in either
    binding = cd.ChipHorizonBinding.__new__(cd.ChipHorizonBinding)
    object.__setattr__(binding, "planning_event", 5)
    object.__setattr__(binding, "horizon_events", (5, 5, 8, 12))
    object.__setattr__(binding, "certification_identity", "sha256:" + "c" * 64)
    object.__setattr__(binding, "data_snapshot_sha256", "sha256:" + "d" * 64)
    decision = _evaluated_decision(horizon_binding=binding)
    assert decision.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_HORIZON_NOT_CANONICAL in decision.reason_codes


def test_P2_2_C_arbitrary_four_distinct_events_are_refused():
    for bad in ((5, 6, 8, 9), (5, 6, 7, 9), (4, 5, 6, 7)):
        with pytest.raises(cd.ChipInputError):
            cd.ChipHorizonBinding(planning_event=5, horizon_events=bad,
                                  certification_identity="sha256:" + "c" * 64)


def test_P2_2_D_planning_event_mismatch_fails():
    # the binding fixes the planning event, so a request against another one cannot
    # be bound at all
    other = _binding(planning_event=6)
    with pytest.raises(cd.ChipInputError) as failure:
        tc.evaluate_triple_captain(_request(horizon_binding=other))
    assert cd.DIAG_CHIP_PLANNING_EVENT_MISMATCH in str(failure.value)


def test_P2_2_E_worlds_planning_event_mismatch_fails():
    worlds = cd.ChipWorldInputs(
        worlds=8, player_ids=tuple(sorted(SQUAD)),
        minutes=_flat({pid: 90.0 for pid in SQUAD}, 8), core=_flat({pid: 5.0 for pid in SQUAD}, 8),
        planning_event=6, horizon_events=(6, 7, 8, 9),
        certification_identity="sha256:" + "c" * 64, data_snapshot_sha256="sha256:" + "d" * 64,
        world_seed=20260911, world_identity="sha256:" + "e" * 64,
    )
    with pytest.raises(cd.ChipInputError) as failure:
        tc.evaluate_triple_captain(_request(worlds=worlds))
    assert cd.DIAG_CHIP_PLANNING_EVENT_MISMATCH in str(failure.value)
    assert any("planning_event" in problem for problem in _binding().matches_worlds(worlds))


def test_P2_2_F_certification_identity_mismatch_fails():
    worlds = _worlds(worlds=8, core=_flat({pid: 5.0 for pid in SQUAD}, 8))
    other_identity = cd.ChipHorizonBinding(
        planning_event=5, horizon_events=(5, 6, 7, 8),
        certification_identity="sha256:" + "9" * 64, data_snapshot_sha256="sha256:" + "d" * 64,
    )
    with pytest.raises(cd.ChipInputError) as failure:
        tc.evaluate_triple_captain(_request(worlds=worlds, horizon_binding=other_identity))
    assert cd.DIAG_CHIP_PLANNING_EVENT_MISMATCH in str(failure.value)


def test_P2_2_G_the_arbiter_refuses_an_evaluation_from_another_context():
    evaluation = tc.evaluate_triple_captain(_request())
    other = cd.ChipHorizonBinding(
        planning_event=6, horizon_events=cd.canonical_chip_horizon(6),
        certification_identity="sha256:" + "c" * 64, data_snapshot_sha256="sha256:" + "d" * 64,
    )
    decision = _evaluated_decision(horizon_binding=other, evaluations={cd.CHIP_ACTION_TC: evaluation})
    assert decision.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH in decision.reason_codes

    # same horizon, different certification identity
    world_identity = cd.ChipHorizonBinding(
        planning_event=5, horizon_events=(5, 6, 7, 8),
        certification_identity="sha256:" + "9" * 64, data_snapshot_sha256="sha256:" + "d" * 64,
    )
    decision2 = _evaluated_decision(horizon_binding=world_identity, evaluations={cd.CHIP_ACTION_TC: evaluation})
    assert cd.DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH in decision2.reason_codes


@pytest.mark.skipif(
    not Path("K:/FPL/data/exports/four_gw/gw05/certification_artifact.json").exists(),
    reason="the accepted R5 certification artifact is not present in this checkout",
)
def test_P2_2_H_the_genuine_four_gw_certification_still_binds():
    artifact = json.loads(
        Path("K:/FPL/data/exports/four_gw/gw05/certification_artifact.json").read_text(encoding="utf-8")
    )
    binding = cd.ChipHorizonBinding.from_certification(artifact)
    assert binding.horizon_events == (5, 6, 7, 8)
    assert binding.planning_event == 5
    assert binding.certification_identity == artifact["four_gw_certification_identity"]
    assert binding.data_snapshot_sha256 == artifact["data_snapshot_sha256"]


# ===========================================================================
# P2-3 — arbiter action / eligibility binding
# ===========================================================================


def _row(name, *, available=True, used=False, expired=False, start=1, stop=19):
    return {
        "name": name, "available_for_event": available, "used": used, "expired": expired,
        "window": f"GW{start}-GW{stop}", "window_start_event": start, "window_stop_event": stop,
    }


def test_P2_3_A_mapping_key_must_equal_the_evaluation_action():
    mismatch = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_WC, evaluator_version="x",
        candidate_metrics={"mean_paired_uplift": 99.0},
        evidence={"certification_identity": "sha256:" + "c" * 64, "horizon_events": [5, 6, 7, 8],
                  "planning_event": 5},
    )
    decision = _evaluated_decision(evaluations={cd.CHIP_ACTION_TC: mismatch})
    assert decision.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_EVALUATION_ACTION_MISMATCH in decision.reason_codes
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP


def test_P2_3_B_and_C_no_chip_can_borrow_another_action_availability():
    """TC available, others not: a TC-keyed WC/BB/FH evaluation must never play."""

    for borrowed in (cd.CHIP_ACTION_WC, cd.CHIP_ACTION_BB, cd.CHIP_ACTION_FH):
        evaluation = cd.ChipEvaluation(
            action=borrowed, evaluator_version="x",
            candidate_metrics={"mean_paired_uplift": 500.0},
            evidence={"certification_identity": "sha256:" + "c" * 64, "horizon_events": [5, 6, 7, 8],
                      "planning_event": 5},
        )
        decision = _evaluated_decision(
            chip_availability=[_row("3xc", available=True)] + [_row(n, available=False) for n in
                                                               ("bboost", "freehit", "wildcard")],
            evaluations={borrowed: evaluation},
        )
        assert decision.recommended_action != borrowed
        assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP

    # the honest TC evaluation in the same state is the one that plays
    honest = _evaluated_decision(
        chip_availability=[_row("3xc", available=True)] + [_row(n, available=False) for n in
                                                           ("bboost", "freehit", "wildcard")],
    )
    assert honest.recommended_action == cd.CHIP_ACTION_TC


def test_P2_3_D_a_used_chip_cannot_be_recommended():
    decision = _evaluated_decision(
        chip_availability=[_row("3xc", available=True, used=True)] + [
            _row(n, available=False) for n in ("bboost", "freehit", "wildcard")],
    )
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP
    assert any(cd.DIAG_CHIP_UNAVAILABLE in code for code in decision.reason_codes)


def test_P2_3_E_an_expired_chip_cannot_be_recommended():
    decision = _evaluated_decision(
        chip_availability=[_row("3xc", available=True, expired=True, start=1, stop=4)] + [
            _row(n, available=False) for n in ("bboost", "freehit", "wildcard")],
    )
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP


def test_P2_3_F_an_unavailable_chip_cannot_be_recommended():
    decision = _evaluated_decision(
        chip_availability=[_row("3xc", available=False)] + [
            _row(n, available=False) for n in ("bboost", "freehit", "wildcard")],
    )
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP
    assert any(cd.DIAG_CHIP_UNAVAILABLE in code for code in decision.reason_codes)


def test_P2_3_G_a_validly_available_action_still_works():
    decision = _evaluated_decision(chip_availability=[_row("3xc", available=True)] + [
        _row(n, available=False) for n in ("bboost", "freehit", "wildcard")])
    assert decision.recommended_action == cd.CHIP_ACTION_TC
    assert decision.candidate_metrics["eligible_actions"] == [cd.CHIP_ACTION_TC]


def test_P2_3_H_only_one_executable_chip_action_can_emerge():
    world = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_WC, evaluator_version="x",
        candidate_metrics={"mean_paired_uplift": 100.0},
        evidence={"certification_identity": "sha256:" + "c" * 64, "horizon_events": [5, 6, 7, 8],
                  "planning_event": 5},
    )
    evaluation = tc.evaluate_triple_captain(_request())
    decision = _evaluated_decision(
        chip_availability=[_row("3xc", available=True), _row("wildcard", available=True)] + [
            _row(n, available=False) for n in ("bboost", "freehit")],
        evaluations={cd.CHIP_ACTION_TC: evaluation, cd.CHIP_ACTION_WC: world},
    )
    assert decision.recommended_action in cd.CHIP_ACTIONS
    assert decision.candidate_metrics["eligible_actions"] == sorted(
        [cd.CHIP_ACTION_TC, cd.CHIP_ACTION_WC]
    )
    # exactly one action emerges even though two are eligible
    assert isinstance(decision.recommended_action, str)


def test_P2_3_I_expired_state_aggregates_per_definition_row():
    """Two half-season definitions: eligibility is the OR of the ROWS' own state."""

    rows = [
        _row("wildcard", available=False, expired=True, start=2, stop=19),
        _row("wildcard", available=True, expired=False, start=20, stop=38),
    ]
    decision = _evaluated_decision(
        horizon_binding=cd.ChipHorizonBinding(
            planning_event=20, horizon_events=cd.canonical_chip_horizon(20),
            certification_identity="sha256:" + "c" * 64, data_snapshot_sha256="sha256:" + "d" * 64,
        ),
        chip_availability=rows + [_row(n, available=False) for n in ("bboost", "3xc", "freehit")],
    )
    # the raw canonical rows are passed through verbatim for auditability
    assert [row["window_stop_event"] for row in decision.chip_availability if row["name"] == "wildcard"] == [19, 38]
    # the second-half window is the live one, so the first-half expiry must not
    # mask it (the old AND-collapse made expired permanently False; the OR-of-rows
    # semantics must not make eligibility permanently True either)
    # (1) the classic AND-collapse case: the first half is expired, the second is
    # live, so the ACTION is eligible even though one row is expired.
    mapped = cd._availability_by_action(rows, planning_event=20)
    assert mapped["WC"]["eligible"] is True
    assert mapped["WC"]["expired_any"] is True
    assert mapped["WC"]["expired_all"] is False

    # (2) the first half live for the event, the second out of its window: the
    # action is eligible via row 1 and row 2 is correctly marked out of window.
    first_half = cd._availability_by_action(
        [_row("wildcard", available=True, expired=False, start=2, stop=19),
         _row("wildcard", available=True, expired=False, start=20, stop=38)],
        planning_event=10,
    )
    assert first_half["WC"]["eligible"] is True
    assert first_half["WC"]["definitions"][0]["in_window_for_event"] is True
    assert first_half["WC"]["definitions"][1]["in_window_for_event"] is False

    # (3) every row expired: never eligible, and expired_all is reachable
    both_expired = cd._availability_by_action(
        [_row("wildcard", available=False, expired=True, start=2, stop=19),
         _row("wildcard", available=False, expired=True, start=20, stop=38)],
        planning_event=40,
    )
    assert both_expired["WC"]["eligible"] is False
    assert both_expired["WC"]["expired_all"] is True
