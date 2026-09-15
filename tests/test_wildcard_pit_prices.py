"""Finding A — point-in-time canonical market prices.

The price a Wildcard candidate is valued at must come from the canonical
official snapshot valid AT OR BEFORE the decision cutoff.  Taking the latest
snapshot regardless of cutoff would leak post-cutoff information into a
historical replay, and letting a caller supply the price would make
affordability caller-controlled.
"""

from __future__ import annotations

import sqlite3

import pytest

from fpl_brain import database
from fpl_brain import wildcard_request_adapter as ad

CUTOFF = "2026-09-12T19:20:00Z"
BEFORE = "2026-09-10T08:00:00Z"
AFTER = "2026-09-14T08:00:00Z"


@pytest.fixture()
def snapshot_db(tmp_path):
    """A temp canonical store with one player priced before and after the cutoff."""

    path = tmp_path / "snap.db"
    conn = database.connect_database(str(path))
    # parent rows the snapshot rows reference
    # player_snapshots is unique on (player_id, fetch_run_id), so each capture
    # needs its own run.
    for run_id in (1, 2, 3):
        conn.execute(
            "INSERT INTO fetch_runs (id, started_at, finished_at, status, trigger) VALUES (?,?,?,?,?)",
            (run_id, BEFORE, BEFORE, "success", "manual"),
        )
    for player_id in (7, 8):
        conn.execute(
            "INSERT INTO players (id, web_name, first_seen_at, last_seen_at, is_active, raw_json, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (player_id, f"p{player_id}", BEFORE, BEFORE, 1, "{}", BEFORE),
        )
    rows = [
        (7, BEFORE, 70),   # at or before the cutoff
        (7, AFTER, 75),    # after it -- must never be selected for this cutoff
        (8, BEFORE, 55),
    ]
    for index, (player_id, captured_at, cost) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO player_snapshots (player_id, fetch_run_id, captured_at, now_cost, raw_json) "
            "VALUES (?,?,?,?,?)",
            (player_id, index, captured_at, cost, "{}"),
        )
    conn.commit()
    yield conn
    conn.close()


def test_price_A1_a_historical_replay_uses_the_pre_cutoff_price(snapshot_db):
    prices = ad._canonical_market_prices(snapshot_db, [7, 8], CUTOFF)
    assert prices[7] == 70, "must use the price valid at or before the cutoff, not the latest"
    assert prices[8] == 55
    assert prices[7] != 75


def test_price_A2_a_later_cutoff_selects_the_later_row(snapshot_db):
    """Current and historical replay select DIFFERENT canonical rows."""

    later = "2026-09-15T00:00:00Z"
    assert ad._canonical_market_prices(snapshot_db, [7], CUTOFF)[7] == 70
    assert ad._canonical_market_prices(snapshot_db, [7], later)[7] == 75


def test_price_A3_a_player_with_no_pre_cutoff_snapshot_has_no_price(snapshot_db):
    """Missing means missing -- never defaulted to zero or to a later price."""

    early = "2026-09-09T00:00:00Z"
    prices = ad._canonical_market_prices(snapshot_db, [7, 8], early)
    assert 7 not in prices and 8 not in prices


def test_price_A4_caller_price_disagreement_refuses(tmp_path):
    """A caller-supplied price that is not the canonical one refuses."""

    from tests.test_wildcard_request_adapter import (
        _certified, _chip_rows, _legal_owned, _pool, _route,
    )

    players = _pool()
    owned = _legal_owned(players)
    # the caller's row says 10 where the canonical store says 100
    liar = int(sorted(players)[0])
    certified = _certified(players)
    patched = dict(players)
    patched[liar] = ad.wc.WildcardPlayer(
        player_id=liar, position=players[liar].position, club_id=players[liar].club_id,
        market_price_tenths=10, events=dict(players[liar].events), web_name="liar",
    )
    certified = ad.WildcardCertifiedInputs(
        horizon=certified.horizon, chip_horizon_binding=certified.chip_horizon_binding,
        value_horizon_binding=certified.value_horizon_binding,
        players=patched, worlds_by_event=certified.worlds_by_event,
        generation=certified.generation,
    )
    manager = ad.WildcardManagerState(
        entry_id=1, planning_event=5, squad_ids=tuple(sorted(owned)), bank_tenths=30,
        purchase_price_tenths={int(p): players[int(p)].market_price_tenths for p in owned},
        event_start_free_transfers=2, chip_availability=tuple(_chip_rows(5)),
        market_price_tenths={int(pid): int(pl.market_price_tenths) for pid, pl in players.items()},
    )
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(manager, certified, _route(players, owned),
                                  rules=ad.sr.SeasonRules(season="2026/27"),
                                  data_snapshot_sha256="sha256:" + "d" * 64)
    assert "canonical point-in-time snapshot" in str(exc.value)


def test_price_A5_a_non_positive_or_non_integer_price_refuses():
    from tests.test_wildcard_request_adapter import _legal_owned, _manager, _pool

    players = _pool()
    owned = _legal_owned(players)
    prices = {int(pid): int(pl.market_price_tenths) for pid, pl in players.items()}
    prices[int(owned[0])] = -5
    manager = _manager(players, owned, market_price_tenths=prices)
    assert any("non-positive" in p for p in manager.problems())


def test_price_A6_a_missing_canonical_price_refuses():
    from tests.test_wildcard_request_adapter import _legal_owned, _manager, _pool

    players = _pool()
    owned = _legal_owned(players)
    prices = {int(pid): int(pl.market_price_tenths) for pid, pl in players.items()}
    prices.pop(int(owned[0]))
    manager = _manager(players, owned, market_price_tenths=prices)
    assert any("no canonical market price" in p for p in manager.problems())
