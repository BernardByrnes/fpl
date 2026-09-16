"""Fixture-level proofs: real canonical routes that the accepted evaluator accepts.

These run BEFORE any production wiring, so a failing fixture is never mistaken
for a failing converter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import free_hit_route_fixtures as fx  # noqa: E402
from fpl_brain import season_rules as sr  # noqa: E402
from fpl_brain import transfer_state as ts  # noqa: E402

RULES = sr.SeasonRules(season="2026/27")
H1, H2, H3, H4 = 5, 6, 7, 8
EVENTS = (H1, H2, H3, H4)


def _state_of(partial, index: int):
    return partial.actions[index]["transition"].next_event_state


# ---------------------------------------------------------------------------
# SAVE — the canonical normal H1-H4 route
# ---------------------------------------------------------------------------


def test_save_route_no_transfer_is_structurally_valid_and_evaluates():
    start = fx.start_state(event=H1, free_transfers=2)
    partial, terminal = fx.build_canonical_route(start=start, events=EVENTS, transfers={})
    assert [a["kind"] for a in partial.actions] == ["ROLL"] * 4
    assert [a["hit_points"] for a in partial.actions] == [0, 0, 0, 0]
    # A roll still progresses FT canonically: 2 -> 3 -> 4 -> 5 -> 5 (capped).
    assert [_state_of(partial, i).free_transfers for i in range(4)] == [3, 4, 5, 5]
    evaluation = fx.evaluate_route(partial, events=EVENTS)
    assert len(evaluation["per_event"]) == 4
    assert evaluation["cumulative_hits"] == 0


def test_save_route_with_one_beneficial_h1_transfer_changes_h2():
    """Sol's counterexample, built from a REAL transition rather than hand-written."""

    start = fx.start_state(event=H1, free_transfers=2)
    # 12 sells at 50 and 22 costs 40, so the bank genuinely moves.
    partial, terminal = fx.build_canonical_route(
        start=start, events=EVENTS, transfers={H1: ((12, 22),)}, prices={H1: {22: 40}},
    )
    first = partial.actions[0]
    assert first["kind"] == "NORMAL_TRANSFER"
    assert int(first["hit_points"]) == 0, "2 FT pays for one transfer"

    before = first["transition"].before_state
    after = first["transition"].next_event_state
    # THE H2 STATE ARISES FROM THE TRANSITION, not from a synthesised object.
    assert 12 in [p.player_id for p in before.players]
    assert 12 not in [p.player_id for p in after.players]
    assert 22 in [p.player_id for p in after.players]
    assert int(after.bank_tenths) == int(before.bank_tenths) + 10, "50 in, 40 out"
    basis = {p.player_id: p.purchase_price_tenths for p in after.players}
    assert basis[22] == 40, "the bought player carries a canonical basis"
    assert int(after.free_transfers) == 2, "one of two free transfers is spent"

    evaluation = fx.evaluate_route(partial, events=EVENTS)
    assert len(evaluation["per_event"]) == 4
    assert all(row["mean_net_core"] == row["mean_gross_core"] for row in evaluation["per_event"])


def test_save_route_multi_transfer_with_a_hit_counts_the_hit_once():
    """A transfer beyond the bank costs exactly one -4, and it is netted once."""

    start = fx.start_state(event=H1, free_transfers=1)
    partial, _terminal = fx.build_canonical_route(
        start=start, events=EVENTS, transfers={H1: ((12, 22), (11, 21))},
    )
    first = partial.actions[0]
    assert first["kind"] == "NORMAL_TRANSFER"
    assert int(first["hit_points"]) == 4, "one paid transfer beyond 1 FT"
    evaluation = fx.evaluate_route(partial, events=EVENTS)
    row = evaluation["per_event"][0]
    assert row["hit_points"] == 4
    assert row["mean_net_core"] == pytest.approx(row["mean_gross_core"] - 4.0)
    assert evaluation["cumulative_hits"] == 4
    assert evaluation["net_core"] == pytest.approx(evaluation["gross_core"] - 4.0)


def test_save_route_handles_a_legitimately_negative_event_value():
    """Canonical values are copied exactly; they are never clamped."""

    start = fx.start_state(event=H1, free_transfers=2)
    partial, _terminal = fx.build_canonical_route(
        start=start, events=EVENTS, transfers={H1: ((12, 22),), H3: ((11, 21),)},
    )
    worlds = {event: fx.world_matrix(event=event, scores={pid: -0.5 for pid in fx.UNIVERSE})
              for event in EVENTS}
    evaluation = fx.evaluate_route(partial, events=EVENTS, worlds=worlds)
    assert all(row["mean_gross_core"] < 0.0 for row in evaluation["per_event"])


# ---------------------------------------------------------------------------
# PLAY — the H2-H4 tail from the RESTORED permanent state
# ---------------------------------------------------------------------------


def test_play_tail_route_from_the_restored_state_is_structurally_valid_and_evaluates():
    """The restored state is the PRE-FH permanent fifteen, and the tail runs H2-H4."""

    restored = fx.start_state(event=H2, squad_ids=fx.SQUAD_IDS, bank_tenths=7, free_transfers=2)
    partial, terminal = fx.build_canonical_route(
        start=restored, events=(H2, H3, H4), transfers={H3: ((12, 22),)},
    )
    assert [a["event"] for a in partial.actions] == [H2, H3, H4]
    evaluation = fx.evaluate_route(partial, events=(H2, H3, H4))
    assert [row["event"] for row in evaluation["per_event"]] == [H2, H3, H4]
    assert int(terminal.bank_tenths) == int(_state_of(partial, 2).bank_tenths)


def test_the_temporary_free_hit_squad_never_appears_in_the_play_tail():
    """A tail built from the permanent fifteen contains no temporary player."""

    temporary = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 12, 13, 14)
    restored = fx.start_state(event=H2, squad_ids=fx.SQUAD_IDS, bank_tenths=7, free_transfers=2)
    partial, _terminal = fx.build_canonical_route(start=restored, events=(H2, H3, H4))
    tail_ids = {int(p.player_id) for p in partial.state.players}
    assert tail_ids == set(fx.SQUAD_IDS)
    assert not (tail_ids & (set(temporary) - set(fx.SQUAD_IDS)))


# ---------------------------------------------------------------------------
# CONTINUITY — the fixture proves the canonical chain really chains
# ---------------------------------------------------------------------------


def test_inter_event_continuity_is_real_not_assumed():
    """Each event's before_state IS the previous event's next_event_state."""

    start = fx.start_state(event=H1, free_transfers=2, bank_tenths=10)
    partial, _terminal = fx.build_canonical_route(
        start=start, events=EVENTS, transfers={H1: ((12, 22),), H3: ((11, 21),)},
    )
    previous = None
    for index, action in enumerate(partial.actions):
        before = action["transition"].before_state
        if previous is not None:
            assert [p.player_id for p in before.players] == [p.player_id for p in previous.players]
            assert int(before.bank_tenths) == int(previous.bank_tenths)
            assert int(before.free_transfers) == int(previous.free_transfers)
            assert ({p.player_id: p.purchase_price_tenths for p in before.players}
                    == {p.player_id: p.purchase_price_tenths for p in previous.players})
        previous = action["transition"].next_event_state
    assert [p.player_id for p in partial.state.players] == [p.player_id for p in previous.players]
    assert int(partial.state.bank_tenths) == int(previous.bank_tenths)


def test_an_illegal_transition_is_rejected_by_the_canonical_engine():
    """The fixture cannot build an illegal route — the engine refuses it."""

    start = fx.start_state(event=H1, free_transfers=2)
    with pytest.raises(AssertionError) as caught:
        # Selling a player the manager does not own.
        fx.build_canonical_route(start=start, events=(H1,), transfers={H1: ((99, 22),)})
    assert "transition rejected" in str(caught.value)
