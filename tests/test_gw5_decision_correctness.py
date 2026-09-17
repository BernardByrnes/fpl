"""P2-02: ONE authoritative captain-value definition, including expected bonus.

The GW5 defect: the armband objective ranked players by CORE while the FPL
captain rule multiplies the player's ACTUAL points, which include bonus.  In the
real GW5 run that left Haaland (core 4.0704, deterministic bonus 0.82) only 0.019
ahead of B.Fernandes (core 4.0506, bonus 0.33) -- a margin indistinguishable from
world-to-world variation -- when the value actually multiplied differs by 0.48.

Bonus is NOT sampled by the accepted Monte Carlo kernel: ``distribution_basis`` is
CORE with bonus deterministic, and the simulator reads it as the analytic "bonus"
component, which is the xPts payload's own ``bonus_xpts``.  It is therefore one
constant per player, and it enters a conditional expectation exactly once.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import manager_lineup as ml
from fpl_brain import manager_worlds as mw

LIVE_DB = Path("K:/FPL/fpl.db")


def _series(value: float, worlds: int = 50) -> list[float]:
    return [float(value)] * worlds


def _two_player_matrix(*, bonus: bool) -> dict:
    """Player A: core 4.10, bonus 0.00.  Player B: core 4.00, bonus 0.80.

    Both appear in every world, so the armband decision turns purely on the value
    the multiplier applies.  Everything else is equal.
    """

    matrix = {
        "worlds": 50,
        "player_ids": [1, 2],
        "minutes": {1: _series(90.0), 2: _series(90.0)},
        "core": {1: _series(4.10), 2: _series(4.00)},
    }
    if bonus:
        matrix["expected_bonus"] = {1: 0.00, 2: 0.80}
    return matrix


# ---------------------------------------------------------------------------
# The discriminator required by the brief
# ---------------------------------------------------------------------------


def test_the_armband_goes_to_the_higher_total_value_not_the_higher_core():
    """core 4.10/bonus 0.00 vs core 4.00/bonus 0.80 -> the second must win."""

    a_terms, _ = mw.captain_terms(_two_player_matrix(bonus=True))
    assert a_terms[2] > a_terms[1], (
        "core 4.00 + 0.80 bonus (4.80 total) must outrank core 4.10 + 0.00 (4.10 total)"
    )
    # The values are exactly the totals the multiplier applies, because both
    # players appear in every world.
    assert a_terms[1] == pytest.approx(4.10)
    assert a_terms[2] == pytest.approx(4.80)


def test_without_a_bonus_block_the_basis_is_declared_core_only():
    """An absent bonus block is a NAMED limitation, never a silent equivalence."""

    matrix = _two_player_matrix(bonus=False)
    a_terms, _ = mw.captain_terms(matrix)
    assert a_terms[1] > a_terms[2], "CORE-only ranking is unchanged from the predecessor"
    assert ml.captain_value_basis(matrix) == ml.CAPTAIN_VALUE_BASIS_CORE_ONLY
    assert ml.captain_value_basis(_two_player_matrix(bonus=True)) == ml.CAPTAIN_VALUE_BASIS_TOTAL


def test_the_bonus_is_scored_once_and_only_when_the_player_appears():
    """No double count, and no bonus for a player who never takes the field."""

    worlds = 50
    absent = {
        "worlds": worlds,
        "player_ids": [1, 2],
        # Player 2 never appears, so his bonus can never be multiplied.
        "minutes": {1: _series(90.0), 2: _series(0.0)},
        "core": {1: _series(4.10), 2: _series(0.00)},
        "expected_bonus": {1: 0.00, 2: 0.80},
    }
    a_terms, _ = mw.captain_terms(absent)
    assert a_terms[2] == pytest.approx(0.0), "an absent player scores no armband value at all"

    always = _two_player_matrix(bonus=True)
    a_terms, c_terms = mw.captain_terms(always)
    # Captain extra = core + bonus exactly once (not twice), vice term = 0 because
    # the captain always appears.
    assert a_terms[2] == pytest.approx(4.00 + 0.80)
    assert c_terms[2][1] == pytest.approx(0.0)


def test_a_vice_bonus_is_also_conditional_on_his_own_appearance():
    """The vice's extra copy carries his own bonus, gated on his own appearance."""

    worlds = 50
    split = {
        "worlds": worlds,
        "player_ids": [1, 2],
        # Captain 1 appears in the first half only; vice 2 appears in the second
        # half only, so he is the armband holder in exactly half the worlds.
        "minutes": {1: [90.0] * 25 + [0.0] * 25, 2: [0.0] * 25 + [90.0] * 25},
        "core": {1: [4.00] * 50, 2: [3.00] * 50},
        "expected_bonus": {1: 0.20, 2: 0.60},
    }
    a_terms, c_terms = mw.captain_terms(split)
    # A[2] = P(2 appears) * (3.00 + 0.60) = 0.5 * 3.60
    assert a_terms[2] == pytest.approx(0.5 * 3.60)
    # C[2][1] = P(2 appears AND 1 absent) * (3.00 + 0.60) = 0.5 * 3.60
    assert c_terms[2][1] == pytest.approx(0.5 * 3.60)


# ---------------------------------------------------------------------------
# Fail-closed structure
# ---------------------------------------------------------------------------


def test_a_malformed_bonus_block_fails_closed():
    """A block that cannot be trusted must raise, never be ignored."""

    template = _two_player_matrix(bonus=True)
    for bad in (
        {1: 0.0},                       # missing a captured player
        {1: 0.0, 2: "not a number"},    # non-numeric
        {1: 0.0, 2: -0.5},              # negative expected bonus
        {1: 0.0, 2: float("nan")},      # non-finite
        [0.0, 0.0],                     # wrong shape entirely
    ):
        matrix = dict(template, expected_bonus=bad)
        with pytest.raises(ml.RouteWorldPlayerMissing):
            ml.validate_world_matrix(matrix, context="test")


def test_a_valid_bonus_block_validates():
    matrix = _two_player_matrix(bonus=True)
    assert ml.validate_world_matrix(matrix, context="test") == 50
    assert ml.expected_bonus_map(matrix) == {1: 0.0, 2: 0.8}
    assert ml.expected_bonus_map(_two_player_matrix(bonus=False)) == {}


# ---------------------------------------------------------------------------
# The armband fallback semantics and the multiplier are UNCHANGED
# ---------------------------------------------------------------------------


def test_captain_vice_fallback_semantics_are_unchanged():
    """captain -> vice -> nobody, on appearance, exactly as before."""

    from fpl_brain.manager_lineup import ManagerPolicy

    policy = ManagerPolicy(
        starter_ids=(1, 2, 3), bench_gk_id=4, bench_outfield_order=(5, 6, 7),
        captain_id=1, vice_captain_id=2,
    )
    core = {1: 4.10, 2: 4.00, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}

    # Captain appears -> captain keeps the armband.
    extra, source = ml.captain_multiplier(policy, {1: 90.0, 2: 90.0}, core)
    assert (extra, source) == (4.10, "CAPTAIN")
    # Captain zero minutes -> the vice takes it.
    extra, source = ml.captain_multiplier(policy, {1: 0.0, 2: 90.0}, core)
    assert (extra, source) == (4.00, "VICE")
    # Neither appears -> NO arbitrary third captain.
    extra, source = ml.captain_multiplier(policy, {1: 0.0, 2: 0.0, 3: 90.0}, core)
    assert (extra, source) == (0.0, "NONE")


def test_the_bonus_does_not_reach_the_per_world_multiplier_used_by_chips():
    """The chip-shared armband authority is deliberately untouched.

    ``captain_multiplier`` is the single armband authority that Triple Captain,
    Bench Boost, Free Hit and Wildcard also call.  Its signature takes no bonus
    map, so no chip's semantics can drift as a side effect of this repair.
    """

    import inspect

    parameters = set(inspect.signature(ml.captain_multiplier).parameters)
    assert parameters == {"policy", "minutes", "core", "require_player_ids"}


# ---------------------------------------------------------------------------
# The data source: ONE definition of deterministic bonus, shared with the MC
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LIVE_DB.exists(), reason="the live database is not present")
def test_the_xpts_bonus_is_the_same_quantity_the_simulator_reads():
    """sum(bonus_xpts) over a player's fixtures == the MC's own deterministic bonus.

    The simulator reads bonus as the analytic "bonus" component, which the
    ``mc_components`` mapping sends to the xPts payload's ``bonus_xpts``.  Reading
    it from the certified xPts run therefore reuses that ONE definition instead of
    introducing a second one.
    """

    conn = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # The canonical GW5 certified pair: xPts run 380, Monte Carlo run 381.
        from_xpts = mw.expected_bonus_by_player(conn, xpts_run_id=380, event=5)
        from_mc: dict[int, float] = {}
        for row in conn.execute(
            "SELECT player_id, payload_json FROM monte_carlo_distributions "
            "WHERE projection_run_id=381 AND event=5"
        ):
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
            value = payload.get("bonus_mean_deterministic")
            if value is None:
                continue
            from_mc[int(row["player_id"])] = from_mc.get(int(row["player_id"]), 0.0) + float(value)

        assert from_xpts, "the certified xPts run must carry bonus_xpts"
        assert set(from_xpts) == set(from_mc)
        mismatches = {
            pid: (from_xpts[pid], from_mc[pid])
            for pid in from_xpts
            if abs(from_xpts[pid] - from_mc[pid]) > 1e-6
        }
        assert not mismatches, f"{len(mismatches)} player(s) disagree: {list(mismatches.items())[:3]}"
        # And the real GW5 armband margin is the one the brief describes.
        assert from_xpts[411] == pytest.approx(0.82, abs=0.01)   # Haaland
        assert from_xpts[426] == pytest.approx(0.33, abs=0.01)   # B.Fernandes
    finally:
        conn.close()
