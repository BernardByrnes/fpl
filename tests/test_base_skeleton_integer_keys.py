"""Focused contracts for integer-projected autosub appearance keys."""

from __future__ import annotations

from fpl_brain import manager_worlds as mw

STAT_FIELDS = (
    "mean_core_base",
    "p_any_autosub",
    "expected_autosub_points_added",
    "expected_number_of_autosubs",
    "bench_gk_used_probability",
    "bench_slot_1_used_probability",
    "bench_slot_2_used_probability",
    "bench_slot_3_used_probability",
)

FILLERS = tuple(range(10_000, 10_070))
OUTFIELD = tuple(range(20_000, 20_013))
START_GK = 20_013
BENCH_GK = 20_014
ALL_PLAYERS = FILLERS + OUTFIELD + (START_GK, BENCH_GK)
POSITIONS = {
    **{pid: "DEF" for pid in FILLERS},
    **{pid: "DEF" for pid in range(20_000, 20_005)},
    **{pid: "MID" for pid in range(20_005, 20_010)},
    **{pid: "FWD" for pid in range(20_010, 20_013)},
    START_GK: "GKP",
    BENCH_GK: "GKP",
}
STARTERS = tuple(sorted((START_GK, *range(20_000, 20_003), *range(20_005, 20_010),
                         20_010, 20_011)))
BENCH_ORDER_DEF_FIRST = (20_003, 20_012, 20_004)
BENCH_ORDER_FWD_FIRST = (20_012, 20_003, 20_004)


def _core(pid: int) -> float:
    if pid == 20_012:
        return 30.0
    if pid in (20_003, 20_004):
        return 5.0
    if pid == BENCH_GK:
        return 11.0
    return 8.0 + (pid % 7)


def _matrix(absent_by_world: tuple[frozenset[int], ...]) -> dict:
    core = {pid: [] for pid in ALL_PLAYERS}
    minutes = {pid: [] for pid in ALL_PLAYERS}
    for absent in absent_by_world:
        for pid in ALL_PLAYERS:
            if pid in absent:
                minutes[pid].append(0.0)
                core[pid].append(0.0)
            else:
                minutes[pid].append(90.0)
                core[pid].append(_core(pid))
    return {
        "worlds": len(absent_by_world),
        "player_ids": list(ALL_PLAYERS),
        "core": core,
        "minutes": minutes,
    }


def _install_fresh_counters(monkeypatch):
    counters = {name: 0 for name in mw.matrix_memo_stats()}
    monkeypatch.setattr(mw, "_MATRIX_MEMO_STATS", counters)
    monkeypatch.setattr(mw, "_MATRIX_MEMO", {})
    return counters


def _assert_exact_rows(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        for field in STAT_FIELDS:
            assert actual_row[field] == expected_row[field], (
                field, actual_row[field], expected_row[field]
            )


def test_integer_key_ignores_unwatched_high_bit_and_handles_watched_indices_above_63(monkeypatch):
    # The evaluated 15-player squad begins after 70 unrelated matrix players.
    assert min(ALL_PLAYERS.index(pid) for pid in (*STARTERS, BENCH_GK, *BENCH_ORDER_DEF_FIRST)) > 63
    skeletons = [(STARTERS, BENCH_GK, BENCH_ORDER_DEF_FIRST)]
    matrix = _matrix((frozenset({FILLERS[-1]}), frozenset()))
    counters = _install_fresh_counters(monkeypatch)

    expected = mw.base_skeleton_stats_reference(skeletons, POSITIONS, matrix)
    counters.update({name: 0 for name in counters})
    actual = mw.base_skeleton_stats(skeletons, POSITIONS, matrix)

    _assert_exact_rows(actual, expected)
    # The two full masks differ only at an unrelated index (69). The watched
    # high-index squad bits are identical, so the second group is a local-cache hit.
    assert counters["entrant_cache_misses"] == 1
    assert counters["entrant_cache_hits"] == 1


def test_integer_key_keeps_bench_order_goalkeeper_and_skeleton_cache_boundaries(monkeypatch):
    # Four distinct watched patterns: an absent starting forward; that absence plus
    # the first bench defender absent; starting keeper absent with cover present; and
    # both keepers absent. All watched player indices are above 63.
    matrix = _matrix((
        frozenset({20_010}),
        frozenset({20_010, 20_003}),
        frozenset({START_GK}),
        frozenset({START_GK, BENCH_GK}),
    ))
    skeletons = [
        (STARTERS, BENCH_GK, BENCH_ORDER_DEF_FIRST),
        (STARTERS, BENCH_GK, BENCH_ORDER_FWD_FIRST),
    ]
    assert min(ALL_PLAYERS.index(pid) for skeleton in skeletons
               for pid in (*skeleton[0], skeleton[1], *skeleton[2])) > 63
    counters = _install_fresh_counters(monkeypatch)

    expected = mw.base_skeleton_stats_reference(skeletons, POSITIONS, matrix)
    counters.update({name: 0 for name in counters})
    actual = mw.base_skeleton_stats(skeletons, POSITIONS, matrix)

    _assert_exact_rows(actual, expected)
    # Four distinct watched patterns per skeleton. Each skeleton starts with a fresh
    # local dictionary even though both project to the same watched-player set.
    assert counters["entrant_cache_misses"] == 8
    assert counters["entrant_cache_hits"] == 0
    assert actual[0]["bench_gk_used_probability"] == 0.25
    assert actual[1]["bench_gk_used_probability"] == 0.25
    assert actual[0]["bench_slot_1_used_probability"] == 0.25
    assert actual[0]["bench_slot_2_used_probability"] == 0.25
    assert actual[1]["bench_slot_1_used_probability"] == 0.5
    assert actual[1]["bench_slot_2_used_probability"] == 0.0
    assert actual[0]["expected_autosub_points_added"] != actual[1]["expected_autosub_points_added"]
