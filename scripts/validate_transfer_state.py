#!/usr/bin/env python3
"""Phase 7A — validate the transfer state engine on the REAL current manager state.

Read-only: builds a derived RouteState from PlanningContext plus an explicit
official PriceSnapshot, prints every player's purchase / market / canonical
selling price, and reconciles against the canonical stored selling-price
evidence.  No transfer is recommended or executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import manager_worlds, transfer_state as ts
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        entry_id = config.get("fpl_entry_id")
        context = get_planning_context(conn, int(entry_id), int(args.gw), season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        selling = {int(row["player_id"]): row for row in context.selling_prices}

        meta = {}
        prices = {}
        for pid in squad["squad_ids"]:
            player = conn.execute("SELECT id, team_id FROM players WHERE id=?", (pid,)).fetchone()
            meta[pid] = ts.PlayerMeta(pid, squad["positions"][pid], int(player["team_id"]))
            snapshot = conn.execute(
                "SELECT now_cost FROM player_snapshots WHERE player_id=? ORDER BY captured_at DESC LIMIT 1",
                (pid,),
            ).fetchone()
            prices[pid] = int(snapshot["now_cost"]) if snapshot else None

        missing = {pid: price for pid, price in prices.items() if price is None}
        route_players = []
        rows = []
        discrepancies = []
        for pid in squad["squad_ids"]:
            evidence = selling.get(pid, {})
            purchase = evidence.get("purchase_price")
            current = prices.get(pid)
            calculated = ts.selling_price_tenths(int(purchase), int(current)) if purchase is not None and current is not None else None
            stored = evidence.get("effective_selling_price")
            if calculated is not None and stored is not None and int(stored) != int(calculated):
                discrepancies.append({
                    "player_id": pid, "name": squad["names"][pid],
                    "purchase": int(purchase), "current": int(current),
                    "calculated": int(calculated), "stored": int(stored),
                })
            if purchase is not None:
                route_players.append(ts.RoutePlayer(pid, squad["positions"][pid], int(meta[pid].club_id), int(purchase)))
            rows.append({
                "player_id": pid, "name": squad["names"][pid], "position": squad["positions"][pid],
                "club_id": int(meta[pid].club_id), "purchase_price_tenths": purchase,
                "current_market_price_tenths": current, "canonical_selling_price_tenths": calculated,
                "stored_effective_selling_price_tenths": stored,
                "stored_source": evidence.get("source"), "stored_status": evidence.get("status"),
            })

        snapshot = ts.PriceSnapshot(event=int(args.gw), prices={pid: p for pid, p in prices.items() if p is not None})
        manager = context.manager_state or {}
        state = ts.RouteState(
            event=int(args.gw), players=tuple(route_players),
            bank_tenths=int(manager.get("bank") or 0),
            free_transfers=int(manager.get("free_transfers") or 0),
            chip_state=tuple(
                {"name": chip.get("name"), "number": chip.get("number"), "available_for_event": chip.get("available_for_event")}
                for chip in (context.chips or [])
            ),
        )

        report = {
            "phase": "phase7a_transfer_state_validation_v1.0.0",
            "transfer_rules_version": ts.TRANSFER_RULES_VERSION,
            "event": int(args.gw),
            "entry_id": context.entry_id,
            "price_snapshot_id": snapshot.identity(),
            "squad_hash": state.squad_hash(),
            "bank_tenths": state.bank_tenths,
            "free_transfers": state.free_transfers,
            "squad_ids": list(squad["squad_ids"]),
            "position_counts": state.position_counts(),
            "club_counts": {str(k): v for k, v in sorted(state.club_counts().items())},
            "players": rows,
            "missing_prices": missing,
            "selling_price_discrepancies": discrepancies,
            "chip_state_snapshot": list(state.chip_state),
            "no_recommendation": True,
        }
        if args.out:
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"Phase 7A real-state validation: GW{args.gw} entry {context.entry_id}")
        print(f"  squad {len(state.players)} | bank {state.bank_tenths} tenths | FT {state.free_transfers} | snapshot {snapshot.identity()[:24]}…")
        print(f"  positions {state.position_counts()} | max club count {max(state.club_counts().values())}")
        print(f"  | {'player':26s} | pos | club | buy | mkt | calc sell | stored sell |")
        for row in rows:
            print(f"  | {str(row['name'])[:26]:26s} | {row['position']} | {row['club_id']:4d} | "
                  f"{row['purchase_price_tenths']!s:>3s} | {row['current_market_price_tenths']!s:>3s} | "
                  f"{row['canonical_selling_price_tenths']!s:>4s} | {row['stored_effective_selling_price_tenths']!s:>4s} |")
        print(f"  missing prices: {missing}")
        print(f"  selling-price discrepancies: {discrepancies}")
        return 3 if discrepancies else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
