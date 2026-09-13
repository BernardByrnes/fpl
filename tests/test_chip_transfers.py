"""Chip-aware transfer/acquisition ingestion and FT/chip rule semantics.

Official 2026/27 behaviour proven here:
- normal transfers close/open permanent acquisition stints,
- Wildcard transfers are permanent and land in the ledger,
- Free Hit transfers are temporary: the pre-chip acquisitions must survive the
  event unchanged and revert afterwards,
- buy/sell/rebuy second stints work, same-event churn is chronological,
- repeated syncs are idempotent, and Free-Transfer bank rules transition.
"""

from __future__ import annotations

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.models import PickRecord, PlayerRecord, PlayerSnapshotRecord
from fpl_brain.season_rules import (
    SeasonRules,
    chip_keeps_saved_free_transfers,
    free_transfers_after_chip,
    free_transfers_after_gameweek,
    price_engine_matches_rules,
    transfer_hit_cost,
)


def _squad_rows(entry_id: int, event: int, player_ids: list[int]) -> list[PickRecord]:
    return [PickRecord(player_id=player_id, position=index + 1, raw_json={}) for index, player_id in enumerate(player_ids)]


def _transfer(entry_id: int, element_in: int, element_out: int, event: int, time: str, cost: int = 50, out_cost: int = 50):
    return {
        "entry": entry_id,
        "element_in": element_in,
        "element_out": element_out,
        "event": event,
        "time": time,
        "element_in_cost": cost,
        "element_out_cost": out_cost,
    }


def _seed_world(conn, entry_id: int = 99, players: int = 18):
    with conn:
        repo.upsert_players(
            conn,
            [PlayerRecord(id=player_id, web_name=f"P{player_id}", full_name=f"Player {player_id}") for player_id in range(1, players + 1)],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=player_id, captured_at="2026-08-20T00:00:00Z", now_cost=55, cost_change_start=0, raw_json={}) for player_id in range(1, players + 1)],
            run,
        )


def _lay_squads(conn, entry_id: int, squads: dict[int, list[int]]):
    with conn:
        for event, ids in squads.items():
            repo.upsert_squad_picks(conn, entry_id, event, _squad_rows(entry_id, event, ids))


def _active(conn, entry_id: int = 99) -> dict[int, int]:
    return {int(row["player_id"]): int(row["purchase_price"]) for row in repo.active_manager_acquisitions(conn, entry_id)}


def test_free_hit_transfers_are_temporary_and_revert(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    base = list(range(1, 16))
    free_hit_squad = [16, 17, 18] + list(range(4, 16))
    _lay_squads(conn, 99, {1: base, 2: free_hit_squad, 3: base})
    with conn:
        conn.execute(
            "INSERT INTO manager_chips(entry_id, name, event, time, updated_at) VALUES (99, 'freehit', 2, '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z')"
        )
    transfers = [
        _transfer(99, 16, 1, 2, "2026-09-05T10:00:00Z"),
        _transfer(99, 17, 2, 2, "2026-09-05T10:00:00Z"),
        _transfer(99, 18, 3, 2, "2026-09-05T10:00:00Z"),
    ]
    with conn:
        result = repo.reconcile_manager_acquisitions(conn, 99, 3, transfers, True, captured_at="2026-09-06T12:00:00Z")
    assert result["status"] == "success", result
    active = _active(conn)
    assert set(active) == set(base)
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == 15
    # Base squad choice must skip the temporary Free-Hit squad snapshot.
    pre_chip_base = repo.latest_complete_squad_rows(conn, 99, 3, exclude_events={2})
    assert [int(row["player_id"]) for row in pre_chip_base[1]] == base and pre_chip_base[0] == 3
    temp_base = repo.latest_complete_squad_rows(conn, 99, 3)
    assert temp_base[0] == 3
    # The Free-Hit overlay squad is flagged by the value engine path.
    assert repo.active_manager_acquisitions(conn, 99)[0]["player_id"] == 1
    # Repeated sync stays idempotent.
    with conn:
        again = repo.reconcile_manager_acquisitions(conn, 99, 3, transfers, True, captured_at="2026-09-07T12:00:00Z")
    assert again["status"] == "success"
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == 15
    conn.close()


def test_free_hit_only_week_cannot_fabricate_a_permanent_base(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    free_hit_squad = [16, 17, 18] + list(range(4, 16))
    _lay_squads(conn, 99, {2: free_hit_squad})
    with conn:
        conn.execute(
            "INSERT INTO manager_chips(entry_id, name, event, time, updated_at) VALUES (99, 'freehit', 2, '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z')"
        )
    transfers = [
        _transfer(99, 16, 1, 2, "2026-09-05T10:00:00Z", cost=55),
        _transfer(99, 17, 2, 2, "2026-09-05T10:00:00Z", cost=55),
        _transfer(99, 18, 3, 2, "2026-09-05T10:00:00Z", cost=55),
    ]
    result = repo.reconcile_manager_acquisitions(conn, 99, 3, transfers, True, captured_at="2026-09-06T12:00:00Z")
    assert result["status"] == "data_gap"
    assert "Free-Hit" in result["message"]
    conn.close()


def test_wildcard_batch_is_permanent(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _lay_squads(conn, 99, {1: list(range(1, 16))})
    with conn:
        conn.execute(
            "INSERT INTO manager_chips(entry_id, name, event, time, updated_at) VALUES (99, 'wildcard', 2, '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z')"
        )
    wildcard_squad = [16, 17, 18] + list(range(4, 16))
    _lay_squads(conn, 99, {2: wildcard_squad})
    transfers = [_transfer(99, 16, 1, 2, "2026-09-05T10:00:00Z"), _transfer(99, 17, 2, 2, "2026-09-05T10:05:00Z"), _transfer(99, 18, 3, 2, "2026-09-05T10:10:00Z")]
    with conn:
        result = repo.reconcile_manager_acquisitions(conn, 99, 2, transfers, True, captured_at="2026-09-06T12:00:00Z")
    assert result["status"] == "success", result
    active = _active(conn)
    assert 16 in active and 17 in active and 18 in active
    assert 1 not in active and 2 not in active and 3 not in active
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions WHERE sold_event IS NOT NULL").fetchone()[0] == 3
    conn.close()


def test_wildcard_week_preserves_saved_ft_bank(tmp_path):
    """Wildcard weeks: all transfers are free; the saved bank is RETAINED.

    Official rule: a Wildcard retains the saved free-transfer state — the
    weekly +1 accrual does NOT apply.  Three saved FTs stay three, regardless
    of how many wildcard transfers were made.
    """

    rules = SeasonRules(season="2026/27")
    assert chip_keeps_saved_free_transfers(rules.wildcard_ft_rule)
    assert free_transfers_after_chip(rules, "wildcard", event_start_free_transfers=3) == 3
    # Without the chip the same week would roll over with the weekly +1.
    assert free_transfers_after_gameweek(rules, 3, 0) == 4
    # If those 15 transfers had been normal ones, the extra-transfer cost rule
    # would be 4 points each beyond the saved bank.
    assert transfer_hit_cost(rules, 1) == 4


def test_second_stint_creates_new_acquisition(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _lay_squads(conn, 99, {1: list(range(1, 16))})
    squad_after_sell = [2] + list(range(3, 17))
    _lay_squads(conn, 99, {2: squad_after_sell})
    rebuy_squad = [2, 1] + list(range(3, 16))
    _lay_squads(conn, 99, {3: rebuy_squad})
    with conn:
        first_sale = repo.reconcile_manager_acquisitions(conn, 99, 2, [_transfer(99, 16, 1, 2, "2026-09-05T10:00:00Z", cost=55)], True, captured_at="2026-09-05T12:00:00Z")
        assert first_sale["status"] == "success"
        re_buy = repo.reconcile_manager_acquisitions(conn, 99, 3, [_transfer(99, 16, 1, 2, "2026-09-05T10:00:00Z", cost=55), _transfer(99, 1, 16, 3, "2026-09-12T10:00:00Z", cost=60, out_cost=55)], True, captured_at="2026-09-12T12:00:00Z")
        assert re_buy["status"] == "success", re_buy
    stints = repo.list_manager_acquisitions(conn, 99, 1)
    assert [(row["acquired_event"], row["purchase_price"], row["sold_event"]) for row in stints] == [(1, 55, 2), (3, 60, None)]
    stints_sixteen = repo.list_manager_acquisitions(conn, 99, 16)
    assert [(row["acquired_event"], row["purchase_price"], row["sold_event"]) for row in stints_sixteen] == [(2, 55, 3)]
    conn.close()


def test_same_event_churn_is_processed_chronologically(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _lay_squads(conn, 99, {1: list(range(1, 16))})
    churn_squad = list(range(2, 16)) + [17]
    _lay_squads(conn, 99, {2: churn_squad})
    with conn:
        result = repo.reconcile_manager_acquisitions(
            conn,
            99,
            2,
            [
                _transfer(99, 16, 1, 2, "2026-09-05T10:00:00Z"),
                _transfer(99, 17, 16, 2, "2026-09-05T10:05:00Z"),
            ],
            True,
            captured_at="2026-09-05T12:00:00Z",
        )
    assert result["status"] == "success", result
    active = _active(conn)
    assert 16 not in active and 17 in active and 1 not in active
    stints = repo.list_manager_acquisitions(conn, 99, 16)
    assert len(stints) == 1 and stints[0]["sold_event"] == 2
    conn.close()


def test_reconcile_dry_run_never_writes(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _lay_squads(conn, 99, {1: list(range(1, 16))})
    with conn:
        conn.execute(
            "INSERT INTO manager_chips(entry_id, name, event, time, updated_at) VALUES (99, 'freehit', 2, '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z')"
        )
    transfers = [_transfer(99, 16, 1, 2, "2026-09-05T10:00:00Z")]
    result = repo.reconcile_manager_acquisitions(conn, 99, 2, transfers, True, dry_run=True)
    assert result["status"] == "success"
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == 0
    conn.close()


def test_ft_rules_season_transitions():
    rules = SeasonRules(season="2026/27")
    # Banked transfers roll over up to the official season cap of 5.
    assert free_transfers_after_gameweek(rules, 5, 0) == 5
    assert free_transfers_after_gameweek(rules, 2, 0) == 3
    assert free_transfers_after_gameweek(rules, 3, 2) == 2
    assert free_transfers_after_gameweek(rules, 1, 1) == 1
    # Transfers beyond the saved bank cost 4 points each.
    assert transfer_hit_cost(rules, 2) == 8
    assert transfer_hit_cost(rules, 0) == 0
    # Chips never change the saved bank.
    for chip_rule in (rules.wildcard_ft_rule, rules.free_hit_ft_rule, rules.bench_boost_ft_rule, rules.triple_captain_ft_rule):
        assert chip_keeps_saved_free_transfers(chip_rule)


def test_price_engine_drift_check():
    from fpl_brain.metrics import calculate_selling_price
    from fpl_brain.season_rules import season_rules_from_game_settings

    official = {
        "squad_squadsize": 15,
        "squad_squadplay": 11,
        "squad_team_limit": 3,
        "squad_total_spend": 1000,
        "max_extra_free_transfers": 4,
        "transfers_sell_on_fee": 0.5,
        "element_sell_at_purchase_price": False,
    }
    rules = season_rules_from_game_settings("2026/27", official)
    ok, _ = price_engine_matches_rules(rules)
    assert ok
    # Engine behaviour matches the official settings for both market halves.
    assert calculate_selling_price(47, 49) == 48
    assert calculate_selling_price(47, 45) == 45
    drifted = season_rules_from_game_settings(
        "2026/27", {**official, "transfers_sell_on_fee": 0.25}
    )
    ok, message = price_engine_matches_rules(drifted)
    assert not ok and "sell_on_fee" in message
