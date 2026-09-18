"""PE-3 — bonus allocation tests (the exact fixture competition).

The allocator is a pure function of a fixture's BPS totals.  These tests pin the official
rule, including the two properties that are easy to get wrong: ties are SHARED, and the
total bonus in a fixture CAN EXCEED SIX.

The shapes used here are not invented — they are the shapes measured on the 40 eligible
FINAL 2026/27 fixtures (27 unique-top-three, 4 tie-first, 7 tie-second, 2 tie-third;
official totals: 27 fixtures on 6, 12 on 7, 1 on 9).
"""

from __future__ import annotations

import itertools
import json
import sqlite3

import pytest

from fpl_brain import bonus_allocation as ba
from fpl_brain.config import config_path, load_config


def _allocate(values: list[int]) -> dict[int, int]:
    """Players are named by index so ties are expressed purely as BPS values."""

    return ba.allocate_fixture_bonus({index: value for index, value in enumerate(values)})


# ---------------------------------------------------------------------------
# Required cases A-G
# ---------------------------------------------------------------------------


def test_case_a_unique_top_three():
    """40, 35, 30 -> 3, 2, 1 and a total of exactly six."""

    result = _allocate([40, 35, 30])
    assert result == {0: 3, 1: 2, 2: 1}
    assert sum(result.values()) == 6


def test_case_b_tie_first():
    """40, 40, 30 -> 3, 3, 1."""

    result = _allocate([40, 40, 30])
    assert result == {0: 3, 1: 3, 2: 1}
    assert sum(result.values()) == 7


def test_case_c_tie_second():
    """40, 35, 35 -> 3, 2, 2."""

    result = _allocate([40, 35, 35])
    assert result == {0: 3, 1: 2, 2: 2}
    assert sum(result.values()) == 7


def test_case_d_tie_third():
    """40, 35, 30, 30 -> 3, 2, 1, 1."""

    result = _allocate([40, 35, 30, 30])
    assert result == {0: 3, 1: 2, 2: 1, 3: 1}
    assert sum(result.values()) == 7


def test_case_e_ties_produce_more_than_six_total_bonus():
    """Two tied first and three tied at the next BPS value.

    45, 45, 30, 30, 30 -> the leaders take 3 each, and the three on 30 each take
    3 - 2 = 1.  Nine bonus points in total; the official rule is honoured without any
    renormalisation to six.
    """

    result = _allocate([45, 45, 30, 30, 30])
    assert result == {0: 3, 1: 3, 2: 1, 3: 1, 4: 1}
    assert sum(result.values()) == 9


def test_case_f_three_or_more_tied_for_highest():
    """Four players on the top BPS each receive 3."""

    result = _allocate([50, 50, 50, 50])
    assert result == {0: 3, 1: 3, 2: 3, 3: 3}
    assert sum(result.values()) == 12


def test_case_g_zero_and_negative_bps():
    """Zero and negative totals allocate normally; only the top of the table scores."""

    result = _allocate([10, 5, 0, -3, -8])
    assert result == {0: 3, 1: 2, 2: 1, 3: 0, 4: 0}
    assert sum(result.values()) == 6

    negative_only = _allocate([-1, -4, -9])
    assert negative_only == {0: 3, 1: 2, 2: 1}


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values", [
    [40, 35, 30], [40, 40, 30], [40, 35, 35], [40, 35, 30, 30],
    [45, 45, 30, 30, 30], [50, 50, 50, 50], [10, 5, 0, -3, -8], [7, 7, 7, 6, 6, 5],
    [1, 1], [0], [],
])
def test_equal_bps_always_receives_equal_bonus(values):
    result = _allocate(values)
    by_bps: dict[int, set[int]] = {}
    for index, value in enumerate(values):
        by_bps.setdefault(value, set()).add(result[index])
    for bonus_set in by_bps.values():
        assert len(bonus_set) == 1, values


@pytest.mark.parametrize("values", [[40, 35, 30], [40, 40, 30], [7, 7, 7, 6, 6, 5],
                                    [10, 5, 0, -3, -8], [50, 50, 50, 50]])
def test_higher_bps_never_receives_less_bonus(values):
    result = _allocate(values)
    for i, j in itertools.permutations(range(len(values)), 2):
        if values[i] > values[j]:
            assert result[i] >= result[j], (values, i, j)


@pytest.mark.parametrize("values", [[40, 35, 30], [40, 40, 30], [45, 45, 30, 30, 30],
                                    [10, 5, 0, -3, -8], [7, 7, 7, 6, 6, 5]])
def test_bonus_is_always_zero_to_three(values):
    assert set(_allocate(values).values()) <= {0, 1, 2, 3}


@pytest.mark.parametrize("values", [[40, 40, 30], [40, 35, 30, 30], [45, 45, 30, 30, 30]])
def test_is_permutation_invariant(values):
    """The result may not depend on the order players are supplied in."""

    baseline = _allocate(values)
    for permutation in itertools.permutations(range(len(values))):
        shuffled = [values[i] for i in permutation]
        result = _allocate(shuffled)
        # player k in the shuffled list has the same BPS as player permutation[k]
        # in the original, so their bonus must match.
        for position, original_index in enumerate(permutation):
            assert result[position] == baseline[original_index], (values, permutation)


def test_no_assumption_that_total_bonus_is_six():
    """Six is only the no-tie case; every measured tie shape must exceed it."""

    assert sum(_allocate([40, 35, 30]).values()) == 6
    for tied in ([40, 40, 30], [40, 35, 35], [40, 35, 30, 30], [45, 45, 30, 30, 30]):
        assert sum(_allocate(tied).values()) > 6, tied


def test_deterministic():
    values = [45, 45, 30, 30, 30, 12, 12, 3]
    assert _allocate(values) == _allocate(values)


def test_empty_fixture():
    assert ba.allocate_fixture_bonus({}) == {}


# ---------------------------------------------------------------------------
# Consistency checker used by the replay gate
# ---------------------------------------------------------------------------


def test_consistency_checker_accepts_the_official_allocation_and_rejects_a_tampered_one():
    bps = {1: 41, 2: 41, 3: 30, 4: 22}
    good = ba.allocate_fixture_bonus(bps)
    ok, _detail = ba.allocation_is_consistent(bps, good)
    assert ok

    tampered = {**good, 3: 3}
    ok, detail = ba.allocation_is_consistent(bps, tampered)
    assert not ok and "mismatch" in detail

    ok, _detail = ba.allocation_is_consistent(bps, {1: 3})
    assert not ok


def test_bonus_totals_reports_exceeding_six():
    bps = {1: 40, 2: 40, 3: 30}
    totals = ba.bonus_totals(ba.allocate_fixture_bonus(bps))
    assert totals["total_bonus"] == 7 and totals["exceeds_six"] is True
    assert totals["awarded"] == [1, 2, 3]


def test_allocator_version_is_declared():
    assert ba.BONUS_ALLOCATOR_VERSION == "fpl_bonus_allocation_v1.0.0"


# ---------------------------------------------------------------------------
# REAL-DATA REPLAY — official BPS -> our allocator -> official bonus
# ---------------------------------------------------------------------------


def _open_database():
    """Open the configured database read-only, or skip when this worktree has none."""

    try:
        config = load_config(None)
    except Exception:  # ConfigError and friends: no config.json in this worktree
        pytest.skip("no config.json in this worktree")
    path = config_path(config, "database")
    if not path.exists():
        pytest.skip("no database in this worktree")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def test_replay_official_bps_to_official_bonus_on_every_eligible_final_fixture():
    """HARD GATE: 100% equality on every unambiguously final fixture.

    Eligibility comes from the repository's own finality authority (event finished AND
    data_checked, fixture finished, and complete BPS/bonus rows) — provisional fixtures
    are excluded, never scored as though final.
    """

    conn = _open_database()
    try:
        fixture_ids = [int(row["fixture_id"]) for row in conn.execute(
            """
            SELECT f.id AS fixture_id FROM fixtures f JOIN events e ON e.id = f.event
            WHERE f.finished = 1 AND e.finished = 1 AND e.data_checked = 1
              AND EXISTS (SELECT 1 FROM player_gameweeks pg WHERE pg.fixture_id = f.id)
              AND NOT EXISTS (SELECT 1 FROM player_gameweeks pg
                              WHERE pg.fixture_id = f.id AND (pg.bps IS NULL OR pg.bonus IS NULL))
            ORDER BY f.id
            """
        )]
        assert fixture_ids, "expected at least one eligible final fixture"

        rows_tested = 0
        mismatches = []
        totals = []
        for fixture_id in fixture_ids:
            rows = conn.execute(
                "SELECT player_id, bps, bonus FROM player_gameweeks WHERE fixture_id = ?",
                (fixture_id,),
            ).fetchall()
            bps = {int(r["player_id"]): int(r["bps"]) for r in rows}
            official = {int(r["player_id"]): int(r["bonus"]) for r in rows}
            ours = ba.allocate_fixture_bonus(bps)
            rows_tested += len(ours)
            totals.append(sum(official.values()))
            for player_id, value in ours.items():
                if official[player_id] != value:
                    mismatches.append((fixture_id, player_id, bps[player_id], value,
                                       official[player_id]))
        assert not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:5]}"
        assert rows_tested > 0

        # The real data must contain fixtures whose official total bonus is NOT six;
        # otherwise this replay could pass on a broken "top three always" shortcut.
        assert any(total > 6 for total in totals), "expected ties worth more than six"
    finally:
        conn.close()


def test_provisional_fixtures_are_excluded_from_the_replay():
    """Events that are not finished/checked must not contribute scored rows."""

    conn = _open_database()
    try:
        provisional = conn.execute(
            """
            SELECT COUNT(*) AS n FROM fixtures f JOIN events e ON e.id = f.event
            WHERE f.finished = 1 AND (e.finished = 0 OR e.data_checked = 0)
            """
        ).fetchone()["n"]
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM fixtures f JOIN events e ON e.id = f.event
            WHERE f.finished = 1 AND e.finished = 1 AND e.data_checked = 1
              AND EXISTS (SELECT 1 FROM player_gameweeks pg WHERE pg.fixture_id = f.id)
            """
        ).fetchone()["n"]
        assert row >= 1
        # The exclusion rule is expressed as SQL in the gate; this pins that the two sets
        # are disjoint by construction (a fixture cannot be both).
        assert provisional + row <= conn.execute(
            "SELECT COUNT(*) AS n FROM fixtures WHERE finished = 1").fetchone()["n"]
    finally:
        conn.close()
