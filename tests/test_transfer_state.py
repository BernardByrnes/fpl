"""Phase 7A — deterministic transfer state engine acceptance tests.

Pure and fast: no database writes, no football simulation, no Phase-5/6 calls.
"""

from __future__ import annotations

import itertools
import random

import pytest

from fpl_brain import transfer_state as ts

# ---------------------------------------------------------------------------
# Synthetic squad: 2 GKP / 5 DEF / 5 MID / 3 FWD, no club above 3 players.
# ---------------------------------------------------------------------------

POSITION = {
    1: "GKP", 2: "GKP",
    11: "DEF", 12: "DEF", 13: "DEF", 14: "DEF", 15: "DEF",
    21: "MID", 22: "MID", 23: "MID", 24: "MID", 25: "MID",
    31: "FWD", 32: "FWD", 33: "FWD",
}
CLUB = {
    1: 1, 2: 2,
    11: 11, 12: 11, 13: 11, 14: 12, 15: 12, 21: 12, 22: 13, 23: 13, 24: 13,
    25: 14, 31: 14, 32: 14, 33: 15,
}
# Incoming pool (unowned).
POOL_POSITION = {
    41: "DEF", 42: "DEF", 43: "DEF", 44: "DEF",
    51: "MID", 52: "MID", 53: "MID", 54: "MID", 55: "MID",
    61: "FWD", 62: "FWD", 71: "GKP",
}
POOL_CLUB = {
    41: 40, 42: 40, 43: 40, 44: 40, 51: 40, 52: 40, 53: 40, 54: 14, 55: 13,
    61: 40, 62: 41, 71: 41,
}
SQUAD_IDS = sorted(POSITION)


def _meta(**overrides):
    meta = {pid: ts.PlayerMeta(pid, POSITION[pid], CLUB[pid]) for pid in SQUAD_IDS}
    meta.update({pid: ts.PlayerMeta(pid, POOL_POSITION[pid], POOL_CLUB[pid]) for pid in POOL_POSITION})
    for pid, club in overrides.items():
        source = POSITION.get(pid, POOL_POSITION.get(pid))
        meta[int(pid)] = ts.PlayerMeta(int(pid), source, int(club))
    return meta


def _state(*, event=4, bank=0, ft=2, purchases=None, transfers_made=0):
    purchases = purchases or {}
    players = tuple(
        ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], int(purchases.get(pid, 50))) for pid in SQUAD_IDS
    )
    return ts.RouteState(event=event, players=players, bank_tenths=bank, free_transfers=ft,
                         transfers_made_this_event=transfers_made)


def _snapshot(prices=None, event=4):
    base = {pid: 50 for pid in SQUAD_IDS}
    base.update({pid: 50 for pid in POOL_POSITION})
    base.update(prices or {})
    return ts.PriceSnapshot(event=event, prices=base)


def _batch(*pairs):
    return ts.TransferBatch(tuple(ts.TransferAction(out, incoming) for out, incoming in pairs))


# ---------------------------------------------------------------------------
# 1-7 Price
# ---------------------------------------------------------------------------


def test_price_unchanged():
    assert ts.selling_price_tenths(50, 50) == 50
    assert ts.selling_price_tenths(47, 47) == 47


def test_price_rise_plus_1():
    assert ts.selling_price_tenths(50, 51) == 50
    assert ts.selling_price_tenths(47, 48) == 47


def test_price_rise_plus_2():
    assert ts.selling_price_tenths(50, 52) == 51
    assert ts.selling_price_tenths(47, 49) == 48


def test_price_rise_plus_3():
    assert ts.selling_price_tenths(50, 53) == 51


def test_price_rise_plus_4():
    assert ts.selling_price_tenths(50, 54) == 52


def test_price_fall():
    assert ts.selling_price_tenths(50, 47) == 47
    assert ts.selling_price_tenths(50, 49) == 49


def test_price_integer_arithmetic_only():
    for purchase in range(40, 61):
        for current in range(40, 61):
            value = ts.selling_price_tenths(purchase, current)
            assert isinstance(value, int)
            expected = current if current <= purchase else purchase + (current - purchase) // 2
            assert value == expected


def test_generic_half_profit_regression():
    # Generic (no named player): bought 4.7, market 4.8 -> sell 4.7; later market 4.9 -> sell 4.8.
    assert ts.selling_price_tenths(47, 48) == 47
    assert ts.selling_price_tenths(47, 49) == 48


# ---------------------------------------------------------------------------
# 8-21 Batch legality
# ---------------------------------------------------------------------------


def test_valid_single_transfer():
    result = ts.apply_transfer_batch(_state(), _batch((11, 41)), _snapshot(), _meta())
    assert result.ok
    ids = [p.player_id for p in result.squad_after.players]
    assert 11 not in ids and 41 in ids and len(ids) == 15


def test_valid_two_transfer_batch():
    result = ts.apply_transfer_batch(_state(), _batch((11, 41), (21, 51)), _snapshot(), _meta())
    assert result.ok and len(result.squad_after.players) == 15


def test_outgoing_not_owned_rejected():
    result = ts.apply_transfer_batch(_state(), _batch((999, 41)), _snapshot(), _meta())
    assert not result.ok and any("OUTGOING_NOT_OWNED" in e for e in result.errors)
    assert result.squad_after is None


def test_incoming_already_owned_rejected():
    result = ts.apply_transfer_batch(_state(), _batch((11, 21)), _snapshot(), _meta())
    assert any("INCOMING_ALREADY_OWNED" in e for e in result.errors)


def test_duplicate_outgoing_rejected():
    result = ts.apply_transfer_batch(_state(), _batch((11, 41), (11, 42)), _snapshot(), _meta())
    assert any("DUPLICATE_OUTGOING" in e for e in result.errors)


def test_duplicate_incoming_rejected():
    result = ts.apply_transfer_batch(_state(), _batch((11, 41), (12, 41)), _snapshot(), _meta())
    assert any("DUPLICATE_INCOMING" in e for e in result.errors)


def test_same_player_out_and_in_rejected():
    result = ts.apply_transfer_batch(_state(), _batch((11, 11)), _snapshot(), _meta())
    assert any("SAME_PLAYER_OUT_AND_IN" in e for e in result.errors)


def test_wrong_position_replacement_rejected():
    result = ts.apply_transfer_batch(_state(), _batch((11, 51)), _snapshot(), _meta())
    assert any("POSITION_MULTISET_MISMATCH" in e for e in result.errors)


def test_position_multiset_mismatch_rejected():
    result = ts.apply_transfer_batch(_state(), _batch((11, 41), (21, 42)), _snapshot(), _meta())
    assert any("POSITION_MULTISET_MISMATCH" in e for e in result.errors)


def test_club_limit_exceeded_rejected():
    # Club 14 already has 3 players (25, 31, 32); bringing in MID 54 (club 14) is a 4th.
    result = ts.apply_transfer_batch(_state(), _batch((22, 54)), _snapshot(), _meta())
    assert any("CLUB_LIMIT_EXCEEDED" in e for e in result.errors)


def test_exactly_three_club_players_accepted():
    # Club 13 has 3 (22, 23, 24); swap MID 22 out for MID 55 (club 13) keeps 3.
    result = ts.apply_transfer_batch(_state(), _batch((22, 55)), _snapshot(), _meta())
    assert result.ok


def test_insufficient_bank_rejected():
    result = ts.apply_transfer_batch(_state(bank=0), _batch((11, 44)), _snapshot({44: 60}), _meta())
    assert any("INSUFFICIENT_BANK" in e for e in result.errors)


def test_exactly_zero_bank_accepted():
    result = ts.apply_transfer_batch(_state(bank=0), _batch((11, 41)), _snapshot({41: 50}), _meta())
    assert result.ok and result.bank_after_tenths == 0


def test_more_than_twenty_transfers_rejected():
    actions = tuple(ts.TransferAction(1000 + i, 2000 + i) for i in range(21))
    result = ts.apply_transfer_batch(_state(), ts.TransferBatch(actions), _snapshot(), _meta())
    assert any("TRANSFER_CAP_EXCEEDED" in e for e in result.errors)


def test_max_twenty_transfers_within_cap_not_capped():
    actions = tuple(ts.TransferAction(1000 + i, 2000 + i) for i in range(20))
    result = ts.apply_transfer_batch(_state(), ts.TransferBatch(actions), _snapshot(), _meta())
    assert not any("TRANSFER_CAP_EXCEEDED" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 22-24 Atomicity
# ---------------------------------------------------------------------------


def test_invalid_batch_leaves_state_unchanged():
    state = _state()
    before = state.squad_hash()
    result = ts.apply_transfer_batch(state, _batch((11, 41), (12, 41)), _snapshot(), _meta())
    assert not result.ok and result.squad_after is None
    assert state.squad_hash() == before and result.before_state is state


def test_order_permutation_yields_identical_result():
    a = ts.apply_transfer_batch(_state(), _batch((11, 41), (21, 51)), _snapshot(), _meta())
    b = ts.apply_transfer_batch(_state(), _batch((21, 51), (11, 41)), _snapshot(), _meta())
    assert a.ok and b.ok
    assert a.squad_after.squad_hash() == b.squad_after.squad_hash()
    assert a.bank_after_tenths == b.bank_after_tenths
    assert a.next_event_state.free_transfers == b.next_event_state.free_transfers


def test_batch_collectively_affordable_regardless_of_sequence():
    # Bank 0; A (bought 50, now 50) sells 50 and B costs 60 (would fail alone);
    # C (bought 50, now 70) sells 60 and D costs 50.  Batch is affordable: 0.
    state = _state(bank=0, ft=2)
    prices = {41: 60, 42: 50, 12: 70}
    result = ts.apply_transfer_batch(state, _batch((11, 41), (12, 42)), _snapshot(prices), _meta())
    assert result.ok and result.bank_after_tenths == 0


# ---------------------------------------------------------------------------
# 25-27 Acquisition ledger
# ---------------------------------------------------------------------------


def test_incoming_purchase_price_is_current_buy_price():
    result = ts.apply_transfer_batch(_state(bank=10), _batch((11, 41)), _snapshot({41: 57}), _meta())
    acquired = next(p for p in result.squad_after.players if p.player_id == 41)
    assert acquired.purchase_price_tenths == 57


def test_later_selling_price_derives_from_new_purchase_price():
    first = ts.apply_transfer_batch(_state(), _batch((11, 41)), _snapshot({41: 47}), _meta())
    later_state = first.next_event_state
    sell_price = ts.selling_price_tenths(
        next(p for p in later_state.players if p.player_id == 41).purchase_price_tenths, 49
    )
    assert sell_price == 48  # 47 -> 48, not derived from the old owner's basis


def test_old_owner_sale_basis_never_leaks():
    # Old owner bought 11 at 40, now 50 -> sells for 45. New owner buys 41 at 50.
    result = ts.apply_transfer_batch(_state(bank=10, purchases={11: 40}), _batch((11, 41)), _snapshot({11: 50, 41: 50}), _meta())
    assert result.sale_proceeds_tenths == 45
    acquired = next(p for p in result.squad_after.players if p.player_id == 41)
    assert acquired.purchase_price_tenths == 50


# ---------------------------------------------------------------------------
# 28-38 FT / hit transition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("before,transfers,expected_next,expected_hit", [
    (1, 0, 2, 0),
    (1, 1, 1, 0),
    (1, 2, 1, 4),
    (2, 0, 3, 0),
    (2, 1, 2, 0),
    (2, 2, 1, 0),
    (2, 3, 1, 4),
    (5, 0, 5, 0),
    (5, 1, 5, 0),
    (5, 5, 1, 0),
    (5, 6, 1, 4),
])
def test_ft_hit_transition_table(before, transfers, expected_next, expected_hit):
    assert ts.next_event_free_transfers(before, transfers) == expected_next
    _, _, hit = ts.hit_points_for(before, transfers)
    assert hit == expected_hit


def test_free_transfers_used_and_paid_count():
    free_used, paid, hit = ts.hit_points_for(2, 3)
    assert (free_used, paid, hit) == (2, 1, 4)


# ---------------------------------------------------------------------------
# 39-43 ROLL
# ---------------------------------------------------------------------------


def test_roll_is_legal_empty_batch():
    result = ts.apply_transfer_batch(_state(ft=2), ts.TransferBatch.roll(), _snapshot(), _meta())
    assert result.ok and len(result.batch) == 0


def test_roll_changes_nothing_and_bumps_ft():
    state = _state(bank=7, ft=2)
    result = ts.apply_transfer_batch(state, ts.TransferBatch.roll(), _snapshot(), _meta())
    assert result.squad_after.squad_hash() == state.squad_hash()
    assert result.bank_after_tenths == 7
    assert [p.purchase_price_tenths for p in result.squad_after.players] == \
        [p.purchase_price_tenths for p in state.players]
    assert result.hit_points == 0
    assert result.next_event_state.free_transfers == 3


# ---------------------------------------------------------------------------
# 44-49 State purity
# ---------------------------------------------------------------------------


def test_route_state_is_immutable_and_deterministic():
    state = _state()
    with pytest.raises(Exception):
        state.bank_tenths = 5  # frozen dataclass
    a = ts.apply_transfer_batch(state, _batch((11, 41)), _snapshot(), _meta())
    b = ts.apply_transfer_batch(state, _batch((11, 41)), _snapshot(), _meta())
    assert a.as_dict() == b.as_dict()
    assert a.next_event_state.squad_hash() == b.next_event_state.squad_hash()


def test_no_database_writes(tmp_path):
    from fpl_brain.database import connect_database
    conn = connect_database(tmp_path / "fpl.db")
    try:
        before = conn.total_changes
        result = ts.apply_transfer_batch(_state(), _batch((11, 41)), _snapshot(), _meta())
        assert result.ok
        assert conn.total_changes == before  # the pure engine never writes
    finally:
        conn.close()


def test_phase5_run71_and_phase6_artifacts_unchanged():
    import hashlib
    try:
        from fpl_brain.config import config_path, load_config
        from fpl_brain.database import connect_database
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    import sqlite3
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT payload_json FROM monte_carlo_distributions WHERE projection_run_id=71 ORDER BY id"
        ).fetchall()
        digest = hashlib.sha256("|".join(r[0] for r in rows).encode()).hexdigest()[:16]
        assert digest == "603003a85330ff54"
        # Later phases may add future prediction runs; run 71 itself must persist unchanged.
        assert conn.execute("SELECT COUNT(*) FROM projection_runs WHERE id=71").fetchone()[0] == 1
    finally:
        conn.close()
    packet_path = config_path(config, "exports_dir") / "manager" / "gw04" / "manager_lineup_packet.json"
    if packet_path.exists():
        assert '"scoring_basis": "CORE"' in packet_path.read_text(encoding="utf-8")


def test_equivalent_action_sets_are_order_independent():
    permutations = list(itertools.permutations([(11, 41), (21, 51), (31, 61)]))
    hashes = set()
    for order in permutations:
        result = ts.apply_transfer_batch(_state(), _batch(*order), _snapshot(), _meta())
        assert result.ok
        hashes.add((result.squad_after.squad_hash(), result.bank_after_tenths))
    assert len(hashes) == 1


# ---------------------------------------------------------------------------
# 50-51 Real context
# ---------------------------------------------------------------------------


def _real_context():
    try:
        from fpl_brain.config import config_path, load_config
        from fpl_brain.planning import get_planning_context
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    import sqlite3
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    entry = config.get("fpl_entry_id")
    context = get_planning_context(conn, int(entry), 4, season=config.get("season"))
    return conn, context


def test_real_current_squad_resolves_from_context():
    from fpl_brain import manager_worlds
    conn, context = _real_context()
    try:
        squad = manager_worlds.resolve_squad(context, conn)
        assert squad["player_count"] == 15
        counts = {}
        for position in squad["positions"].values():
            counts[position] = counts.get(position, 0) + 1
        assert counts == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        # Free transfers are operator state that legitimately changes as GW4
        # transfers are executed, so assert coherence with the stored manual
        # observation instead of a frozen count.
        manual = (context.manager_state or {}).get("manual") or {}
        if manual.get("free_transfers") is not None:
            assert context.manager_state.get("free_transfers") == int(manual["free_transfers"])
        assert 0 <= int(context.manager_state.get("free_transfers") or 0) <= 5
    finally:
        conn.close()


def test_real_current_selling_prices_reconcile_with_stored_evidence():
    conn, context = _real_context()
    try:
        discrepancies = []
        prices = {}
        for entry in context.selling_prices:
            pid = int(entry["player_id"])
            purchase = entry.get("purchase_price")
            current = entry.get("official_market_price")
            stored = entry.get("effective_selling_price")
            if purchase is None or current is None:
                discrepancies.append((pid, "missing purchase/current price", None, stored))
                continue
            prices[pid] = int(current)
            calculated = ts.selling_price_tenths(int(purchase), int(current))
            if stored is None or int(stored) != calculated:
                discrepancies.append((pid, int(purchase), int(current), calculated, stored))
        assert prices, "no sellable squad price evidence"
        assert discrepancies == [], discrepancies
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_property_random_legal_transfers_preserve_invariants():
    random.seed(11)
    positions = {**POSITION, **POOL_POSITION}
    clubs = {**CLUB, **POOL_CLUB}
    squad_ids = list(SQUAD_IDS)
    pool = sorted(POOL_POSITION)
    for _ in range(40):
        # Random same-position swap that keeps clubs legal and bank non-negative.
        out = random.choice(squad_ids)
        candidates = [pid for pid in pool if positions[pid] == positions[out] and pid not in squad_ids]
        if not candidates:
            continue
        incoming = random.choice(candidates)
        state = ts.RouteState(
            event=4,
            players=tuple(ts.RoutePlayer(pid, positions[pid], clubs[pid], 50) for pid in squad_ids),
            bank_tenths=20, free_transfers=random.randint(1, 5),
        )
        result = ts.apply_transfer_batch(state, _batch((out, incoming)), _snapshot(), _meta())
        if not result.ok:
            assert any("CLUB_LIMIT" in e for e in result.errors)
            continue
        after = result.squad_after
        assert len(after.players) == 15 and len({p.player_id for p in after.players}) == 15
        counts = after.position_counts()
        assert {k: counts[k] for k in ("GKP", "DEF", "MID", "FWD")} == ts.POSITION_COMPOSITION
        assert max(after.club_counts().values()) <= ts.SQUAD_TEAM_LIMIT
        assert after.bank_tenths >= 0
        acquired = next(p for p in after.players if p.player_id == incoming)
        assert acquired.purchase_price_tenths == 50
        assert result.hit_points % ts.HIT_POINTS_PER_EXTRA_TRANSFER == 0
        assert 1 <= result.next_event_state.free_transfers <= ts.MAX_STORED_FREE_TRANSFERS
        # immutability of the pre-state
        assert len([p for p in state.players if p.player_id == out]) == 1
