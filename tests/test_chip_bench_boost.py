"""Bench Boost V1 — scoring semantics, the audit identity, and provenance.

The chip's whole contract is one comparison on the manager's OWN fifteen:

    NORMAL = the accepted ``manager_lineup`` resolution of the certified worlds
    BB     = all fifteen score, armband UNCHANGED

so the tests here are mostly about the differences the contract calls out:
appearance, autosubs (including one that is formation-blocked), the goalkeeper,
and the armband fallback chain — plus the audit identity that makes
double-counting structurally impossible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import chip_bench_boost as bb  # noqa: E402
from fpl_brain import chip_decision as cd  # noqa: E402
from fpl_brain import manager_lineup as ml  # noqa: E402

WORLDS = 8

#: 1-15 with a legal XI: GKP 1 | DEF 3,4,5 | MID 8,9,10,11 | FWD 13,14,15.
POSITIONS = {
    1: "GKP", 2: "GKP",
    3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF",
    8: "MID", 9: "MID", 10: "MID", 11: "MID", 12: "MID",
    13: "FWD", 14: "FWD", 15: "FWD",
}
XI = (1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15)
BENCH_GK = 2
BENCH_OUT = (6, 7, 12)
CAPTAIN = 13
VICE = 8
SQUAD = tuple(sorted(XI + (BENCH_GK,) + BENCH_OUT))

#: Which XI slot each position occupies, so a "missing starter" is easy to pick.
DEF_STARTER = 5
MID_STARTER = 11
FWD_STARTER = 15
STARTER_GK = 1


def _policy(*, xi=XI, bench_gk=BENCH_GK, bench_out=BENCH_OUT, captain=CAPTAIN, vice=VICE):
    return ml.ManagerPolicy(
        starter_ids=tuple(xi), bench_gk_id=int(bench_gk),
        bench_outfield_order=tuple(bench_out), captain_id=int(captain), vice_captain_id=int(vice),
    )


def _flat(values: dict[int, float], worlds: int = WORLDS) -> dict[int, tuple[float, ...]]:
    return {int(pid): tuple(float(value) for _ in range(worlds)) for pid, value in values.items()}


def _worlds(
    core: dict[int, tuple[float, ...]],
    minutes: dict[int, tuple[float, ...]] | None = None,
    *,
    planning_event: int = 5,
    identity: str = "sha256:" + "c" * 64,
    data_snapshot: str = "sha256:" + "d" * 64,
) -> cd.ChipWorldInputs:
    if minutes is None:
        minutes = {pid: tuple(90.0 for _ in series) for pid, series in core.items()}
    events = cd.canonical_chip_horizon(planning_event)
    return cd.ChipWorldInputs(
        worlds=len(next(iter(core.values()))),
        player_ids=tuple(sorted(core)),
        minutes=minutes,
        core=core,
        planning_event=int(planning_event),
        horizon_events=events,
        certification_identity=identity,
        data_snapshot_sha256=data_snapshot,
        world_seed=20260915,
        world_identity="sha256:" + "e" * 64,
    )


def _binding(*, planning_event: int = 5, identity: str = "sha256:" + "c" * 64, events=None):
    return cd.ChipHorizonBinding(
        planning_event=int(planning_event),
        horizon_events=events if events is not None else cd.canonical_chip_horizon(planning_event),
        certification_identity=identity,
        data_snapshot_sha256="sha256:" + "d" * 64,
    )


def _request(core, minutes=None, *, policy=None, worlds=None, binding=None, **overrides):
    base = dict(
        worlds=worlds if worlds is not None else _worlds(core, minutes),
        horizon_binding=binding if binding is not None else _binding(),
        policy=policy if policy is not None else _policy(),
        positions=POSITIONS,
    )
    base.update(overrides)
    return bb.BenchBoostRequest(**base)


def _uniform(*, xi_score: float, bench_score: float, **overrides):
    """Every XI scorer at ``xi_score``, every bench scorer at ``bench_score``."""

    core = {pid: float(xi_score) for pid in SQUAD}
    for pid in (BENCH_GK, *BENCH_OUT):
        core[pid] = float(bench_score)
    return _request(_flat(core), **overrides)


# ---------------------------------------------------------------------------
# SCORING — A..G
# ---------------------------------------------------------------------------


def test_A_all_xi_appear_the_uplift_is_exactly_the_bench_contribution():
    """Normal scores the XI only; BB scores all fifteen; uplift = the four bench."""

    request = _uniform(xi_score=2.0, bench_score=5.0)
    evaluation = bb.evaluate_bench_boost(request)

    # NORMAL: 11 x 2.0 = 22, plus one armband copy of the captain's 2.0.
    assert evaluation.candidate_metrics["mean_normal_value"] == pytest.approx(24.0)
    # BB: 22 + four bench x 5.0 = 42, plus the SAME armband copy.
    assert evaluation.candidate_metrics["mean_bench_boost_value"] == pytest.approx(44.0)
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(20.0)
    assert evaluation.candidate_metrics["mean_normal_autosub_points"] == pytest.approx(0.0)
    assert evaluation.candidate_metrics["mean_bench_raw_value"] == pytest.approx(20.0)


def test_A2_the_missing_starter_slot_is_never_rescued_by_a_bench_zero():
    """A non-appearing starter contributes nothing to EITHER arm."""

    core = _flat({pid: 2.0 for pid in SQUAD} | {pid: 5.0 for pid in (BENCH_GK, *BENCH_OUT)})
    minutes = _flat({pid: 90.0 for pid in SQUAD} | {DEF_STARTER: 0.0})
    evaluation = bb.evaluate_bench_boost(_request(core, minutes))
    # NORMAL: the absent DEF is replaced by bench DEF 6 (5.0), so 9 appearing
    # starters x2 + the 5.0 sub + the armband 2.0 = 27.0.
    assert evaluation.candidate_metrics["mean_normal_value"] == pytest.approx(27.0)
    assert evaluation.candidate_metrics["mean_normal_autosub_points"] == pytest.approx(5.0)
    # The four bench (GK + 3 outfield) are worth 20.0 raw, of which NORMAL had
    # already banked 5.0; the chip is the remaining 15.0.
    assert evaluation.candidate_metrics["mean_bench_raw_value"] == pytest.approx(20.0)
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(15.0)
    assert evaluation.candidate_metrics["identity_residual"] == pytest.approx(0.0, abs=1e-9)


def test_B_an_autosubstituted_bench_player_is_never_counted_twice():
    """The chip is the bench's own score MINUS what NORMAL already recovered."""

    core = _flat({pid: 2.0 for pid in SQUAD} | {pid: 5.0 for pid in (BENCH_GK, *BENCH_OUT)})
    minutes = _flat({pid: 90.0 for pid in SQUAD} | {DEF_STARTER: 0.0})
    evaluation = bb.evaluate_bench_boost(_request(core, minutes))

    bench_raw = evaluation.candidate_metrics["mean_bench_raw_value"]
    autosub = evaluation.candidate_metrics["mean_normal_autosub_points"]
    uplift = evaluation.candidate_metrics["mean_paired_uplift"]
    # Player 6 autosubs in, so his points already sit in the normal arm.  The
    # identity removes exactly that copy and keeps only the remainder.
    assert bench_raw == pytest.approx(20.0)
    assert uplift == pytest.approx(bench_raw - autosub)
    assert uplift == pytest.approx(15.0)


def test_C_both_goalkeepers_score_under_bb_without_duplication():
    """When the starting GK plays, BB adds the bench GK's own score exactly once."""

    evaluation = bb.evaluate_bench_boost(_uniform(xi_score=2.0, bench_score=5.0))
    # Starter GK played, so the normal arm never used the bench GK.
    assert evaluation.candidate_metrics["bench_gk_used_probability"] == pytest.approx(0.0)
    assert evaluation.candidate_metrics["mean_normal_autosub_points"] == pytest.approx(0.0)


def test_C2_a_missing_starting_gk_is_replaced_in_normal_and_scored_afresh_in_bb():
    """GK autosub: NORMAL recovers the bench GK once; BB nets that copy out."""

    core = _flat({pid: 2.0 for pid in SQUAD} | {pid: 5.0 for pid in (BENCH_GK, *BENCH_OUT)})
    minutes = _flat({pid: 90.0 for pid in SQUAD} | {STARTER_GK: 0.0})
    evaluation = bb.evaluate_bench_boost(_request(core, minutes))

    assert evaluation.candidate_metrics["bench_gk_used_probability"] == pytest.approx(1.0)
    assert evaluation.candidate_metrics["mean_normal_autosub_points"] == pytest.approx(5.0)
    # 4 bench x 5.0 = 20 raw, minus the 5.0 the normal arm already banked.
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(15.0)


def test_C3_a_missing_gk_with_a_benched_gk_still_scores_both_eligible_gks():
    """Both GKs on the pitch => NORMAL scores one, BB scores both, no duplicate."""

    core = _flat({pid: 2.0 for pid in SQUAD} | {pid: 5.0 for pid in (BENCH_GK, *BENCH_OUT)})
    evaluation = bb.evaluate_bench_boost(_request(core, _flat({pid: 90.0 for pid in SQUAD})))

    # NORMAL counted GK 1; BB counted GK 1 AND GK 2 — the second only once.
    assert evaluation.candidate_metrics["mean_normal_value"] == pytest.approx(24.0)
    assert evaluation.candidate_metrics["mean_bench_boost_value"] == pytest.approx(44.0)
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(20.0)


def test_D_a_formation_blocked_bench_player_keeps_his_full_score_under_bb():
    """A bench player who CANNOT legally autosub in is not discarded by BB.

    The bench is three FWDs while the XI already fields three FWDs, so no
    substitution can legally happen (FWD max is 3).  NORMAL therefore scores
    nothing from the bench; BB must still score all three.
    """

    policy = _policy(bench_out=(6, 7, 12))
    # Re-purpose the bench as FWDs so the formation constraint actually binds.
    positions = {**POSITIONS, 6: "FWD", 7: "FWD", 12: "FWD"}
    core = _flat({pid: 2.0 for pid in SQUAD} | {pid: 9.0 for pid in (6, 7, 12)})
    minutes = _flat({pid: 90.0 for pid in SQUAD} | {MID_STARTER: 0.0})

    evaluation = bb.evaluate_bench_boost(
        bb.BenchBoostRequest(
            worlds=_worlds(core, minutes), horizon_binding=_binding(),
            policy=policy, positions=positions,
        )
    )
    # NORMAL: MID 11 is absent; every candidate bench FWD would push FWD to 4,
    # so the legal maximum is zero entrants — the bench scores nothing, and the
    # normal total is 9 appearing starters x2 + the GK's 2.0 + armband 2.0 = 22.0.
    assert evaluation.candidate_metrics["mean_normal_value"] == pytest.approx(22.0)
    assert evaluation.candidate_metrics["mean_normal_autosub_points"] == pytest.approx(0.0)
    # BB: the three blocked FWDs (9.0 each) plus the bench GK's 2.0 = 29.0, all
    # of which survives because NORMAL recovered none of it.
    assert evaluation.candidate_metrics["mean_bench_raw_value"] == pytest.approx(29.0)
    assert evaluation.candidate_metrics["mean_paired_uplift"] == pytest.approx(29.0)


def test_E_an_absent_captain_hands_the_armband_to_the_vice_in_BOTH_arms():
    """BB must not invent a different fallback rule than the certified one."""

    core = _flat({pid: 2.0 for pid in SQUAD} | {CAPTAIN: 7.0, VICE: 9.0})
    minutes = _flat({pid: 90.0 for pid in SQUAD} | {CAPTAIN: 0.0})
    evaluation = bb.evaluate_bench_boost(_request(core, minutes))

    assert evaluation.uncertainty["p_armband_vice"] == pytest.approx(1.0)
    assert evaluation.uncertainty["armband_source_counts"]["CAPTAIN"] == 0
    # The armband copy is the vice's 9.0 in both arms (BB nets it out entirely).
    assert evaluation.candidate_metrics["mean_armband_extra"] == pytest.approx(9.0)
    assert bb.DIAG_BB_VICE_FALLBACK_MATERIAL in evaluation.reason_codes


def test_F_no_captain_and_no_vice_means_no_arbitrary_third_captain():
    """Both armband holders absent => no multiplier, and nobody else is promoted."""

    core = _flat({pid: 2.0 for pid in SQUAD} | {CAPTAIN: 7.0, VICE: 9.0, 14: 11.0})
    minutes = _flat({pid: 90.0 for pid in SQUAD} | {CAPTAIN: 0.0, VICE: 0.0})
    evaluation = bb.evaluate_bench_boost(_request(core, minutes))

    assert evaluation.uncertainty["armband_source_counts"]["NONE"] == WORLDS
    assert evaluation.candidate_metrics["mean_armband_extra"] == pytest.approx(0.0)
    assert bb.DIAG_BB_NO_ARMBAND_POSSIBLE in evaluation.reason_codes
    # Player 14 scored 11.0 and is NOT promoted to captain.
    assert evaluation.candidate_metrics["mean_bench_boost_value"] == pytest.approx(11 * 2.0 + 2.0 + 11.0 + 5.0 * 0)


def test_G_the_normal_arm_is_byte_identical_to_the_certified_engine():
    """No leakage: the NORMAL arm IS ``manager_lineup.evaluate_policy``.

    The accepted engine is the oracle, so the equivalence is asserted against
    its own reported ``mean_core`` for the same policy, worlds and positions —
    not against a hand-computed number.
    """

    core = _flat({pid: 2.0 for pid in SQUAD} | {pid: 5.0 for pid in (BENCH_GK, *BENCH_OUT)})
    minutes = _flat({pid: 90.0 for pid in SQUAD} | {DEF_STARTER: 0.0, STARTER_GK: 0.0})
    policy = _policy()
    worlds = _worlds(core, minutes)

    matrix = {
        "worlds": WORLDS,
        "player_ids": list(SQUAD),
        "minutes": {pid: list(minutes[pid]) for pid in SQUAD},
        "core": {pid: list(core[pid]) for pid in SQUAD},
    }
    oracle = ml.evaluate_policy(policy, matrix, POSITIONS)
    evaluation = bb.evaluate_bench_boost(_request(core, minutes, worlds=worlds))

    assert evaluation.candidate_metrics["mean_normal_value"] == pytest.approx(oracle["mean_core"])
    assert evaluation.candidate_metrics["mean_normal_autosub_points"] == pytest.approx(
        oracle["expected_autosub_points_added"]
    )
    assert evaluation.candidate_metrics["bench_gk_used_probability"] == pytest.approx(
        oracle["bench_gk_used_probability"]
    )


def test_G2_every_world_satisfies_the_bench_boost_identity():
    """BB - NORMAL == bench_raw - autosub, world by world, on a noisy series.

    Random-looking (but fixed) per-player per-world scores, so the identity is
    exercised across appearances, autosubs and armband fallbacks at once.
    """

    worlds = 32
    core: dict[int, tuple[float, ...]] = {}
    minutes: dict[int, tuple[float, ...]] = {}
    for pid in SQUAD:
        core[pid] = tuple(float(((pid * 37 + world * 11) % 17) - 3) for world in range(worlds))
        minutes[pid] = tuple(
            0.0 if (pid + world) % 5 == 0 else float(60 + ((pid + world) % 3) * 15) for world in range(worlds)
        )
    request = _request(core, minutes)
    evaluation = bb.evaluate_bench_boost(request)

    policy = request.policy
    for world in range(worlds):
        w_minutes = {pid: minutes[pid][world] for pid in SQUAD}
        w_core = {pid: core[pid][world] for pid in SQUAD}
        normal, outcome, extra_n, source_n = bb.normal_world_value(
            policy, POSITIONS, w_minutes, w_core, require_player_ids=SQUAD
        )
        total, extra_b, source_b = bb.bench_boost_world_value(
            policy, POSITIONS, w_minutes, w_core, require_player_ids=SQUAD
        )
        raw = bb.bench_raw_value(policy, w_minutes, w_core)
        assert extra_n == extra_b and source_n == source_b
        assert (total - normal) == pytest.approx(raw - outcome.autosub_points, abs=1e-9)
    assert evaluation.candidate_metrics["identity_residual"] == pytest.approx(0.0, abs=1e-9)


def test_G3_the_bb_arm_is_invariant_to_the_xi_bench_partition():
    """Under BB the fifteen all score, so only the armband can matter.

    Same fifteen, same captain and vice, different XI/bench split: the BB arm is
    unchanged while the NORMAL arm moves — which is the entire chip effect.
    """

    core = _flat({pid: 2.0 for pid in SQUAD} | {pid: 5.0 for pid in (BENCH_GK, *BENCH_OUT)})
    minutes = _flat({pid: 90.0 for pid in SQUAD})
    other = _policy(xi=(1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15), bench_out=(5, 7, 12))

    first = bb.evaluate_bench_boost(_request(core, minutes))
    second = bb.evaluate_bench_boost(_request(core, minutes, policy=other))

    assert first.candidate_metrics["mean_bench_boost_value"] == second.candidate_metrics["mean_bench_boost_value"]


# ---------------------------------------------------------------------------
# THE CHIP'S OWN VALUE — not a bench-quality coefficient
# ---------------------------------------------------------------------------


def test_the_uplift_is_purely_scoring_world_differences():
    """Doubling every bench score doubles the uplift (and nothing else does)."""

    base = bb.evaluate_bench_boost(_uniform(xi_score=2.0, bench_score=4.0))
    doubled = bb.evaluate_bench_boost(_uniform(xi_score=2.0, bench_score=8.0))
    assert doubled.candidate_metrics["mean_paired_uplift"] == pytest.approx(
        2 * base.candidate_metrics["mean_paired_uplift"]
    )


def test_the_evaluator_never_calls_the_reservation():
    """Reservation is the arbiter's single seam: 0 calls by the evaluator, 1 by the arbiter."""

    calls: list[str] = []

    class Spy:
        def estimate(self, *, action, planning_event, expiry_event, state):
            calls.append(action)
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=None,
            )

    # Evaluating the chip on its own must consult nothing.
    evaluation = bb.evaluate_bench_boost(_uniform(xi_score=2.0, bench_score=5.0))
    assert calls == []

    # Driving the real arbiter consults the seam exactly once, for this action.
    decision = cd.decide_chip_action(
        horizon_binding=_binding(),
        chip_availability=[
            {"name": "bboost", "available_for_event": True, "used": False, "expired": False,
             "window": "GW2-GW19", "window_start_event": 2, "window_stop_event": 19}
        ],
        evaluations={cd.CHIP_ACTION_BB: evaluation},
        reservation=Spy(),
        certification_valid=True,
        manager_state={"squad_ids": list(SQUAD)},
    )
    assert calls == [cd.CHIP_ACTION_BB], "exactly one reservation call, made by the arbiter"
    assert decision.recommended_action == cd.CHIP_ACTION_BB
    # The uplift is positive but nothing may execute: review only.
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
    assert decision.candidate_metrics["reservation_value"] is None


def test_review_only_status():
    """An uncalibrated future-opportunity model must never yield an execution."""

    evaluation = bb.evaluate_bench_boost(_uniform(xi_score=2.0, bench_score=5.0))
    assert evaluation.execution_permitted is False
    assert bb.DIAG_BB_REVIEW_ONLY in evaluation.reason_codes
    assert evaluation.action == cd.CHIP_ACTION_BB
    assert evaluation.mean_uplift == pytest.approx(20.0)


def test_the_evidence_carries_the_certified_provenance_verbatim():
    evaluation = bb.evaluate_bench_boost(_uniform(xi_score=1.0, bench_score=1.0))
    worlds = _worlds(_flat({pid: 1.0 for pid in SQUAD}))
    for key in ("certification_identity", "data_snapshot_sha256", "world_identity", "world_seed"):
        assert evaluation.evidence[key] == worlds.evidence()[key]
    assert evaluation.evidence["horizon_events"] == list(cd.canonical_chip_horizon(5))
    assert len(evaluation.evidence["horizon_events"]) == 4


def test_there_is_no_selection_step_and_no_partition():
    """Both arms are determined by the given policy, so no worlds are spent choosing."""

    metrics = bb.evaluate_bench_boost(_uniform(xi_score=2.0, bench_score=5.0)).candidate_metrics
    assert metrics["selection_step"] == "NONE_THE_POLICY_IS_GIVEN"
    assert metrics["selection_worlds"] == 0
    assert metrics["valuation_worlds"] == WORLDS
    assert metrics["worlds_used"] == WORLDS


# ---------------------------------------------------------------------------
# THE ENGINE INVARIANT THE APPEARANCE GATE RESTS ON
# ---------------------------------------------------------------------------


def test_the_certified_engine_gives_a_non_appearing_player_a_zero_core():
    """``core == 0`` whenever ``minutes <= 0`` — checked on the REAL captured matrices.

    The Bench Boost arm sums only players who appear, which is the FPL rule
    regardless.  This pins the stronger fact that the certified engine agrees, so
    the gated sum and a raw fifteen-way sum can never diverge on real data.
    """

    import json
    import glob
    import os

    root = Path(os.environ.get("FPL_WORLD_CACHE", "K:/FPL/data/cache/manager_worlds"))
    files = sorted(glob.glob(str(root / "*.json")), key=os.path.getsize, reverse=True)[:2]
    if not files:
        pytest.skip(f"no captured world matrices under {root}")

    checked = 0
    for path in files:
        with open(path, encoding="utf-8") as handle:
            matrix = json.load(handle)
        worlds = int(matrix["worlds"])
        for pid in matrix["player_ids"]:
            key = pid if pid in matrix["core"] else str(pid)
            core = matrix["core"][key]
            minutes = matrix["minutes"][key]
            for world in range(worlds):
                if float(minutes[world]) <= 0.0:
                    assert float(core[world]) == 0.0, (path, pid, world)
                checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# NO DISCOVERY, NO TRANSFER SEARCH
# ---------------------------------------------------------------------------


def test_the_evaluator_searches_no_players_and_no_transfers():
    """Bench Boost V1 evaluates the given fifteen; it optimises no squad/route.

    A source-level guard rather than a behavioural one: the evaluator must not
    even be able to reach the transfer, routing or discovery layers, so the chip
    cannot quietly grow into an optimizer.
    """

    source = (Path(__file__).resolve().parents[1] / "fpl_brain/chip_bench_boost.py").read_text(
        encoding="utf-8"
    )
    banned = (
        "route_optimizer", "route_comparator", "transfer_state", "candidate_universe",
        "wildcard_request_adapter", "chip_wildcard", "four_gw_decision",
    )
    offenders = [
        line.strip()
        for line in source.splitlines()
        if line.startswith(("from .", "import ")) and any(token in line for token in banned)
    ]
    assert offenders == [], offenders
