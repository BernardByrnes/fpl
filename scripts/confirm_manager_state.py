#!/usr/bin/env python3
"""Record a user-confirmed manager state (explicit override) for one Gameweek.

The public FPL API can lag behind reality in the final pre-deadline window: a
manager may have already executed transfers the official endpoints have not yet
published.  This CLI is the repository's intended manual tier for that case:

* it writes NO official rows and never rewrites historical snapshots;
* it records the confirmed state as an explicit override with manual
  provenance (`manager_manual_state` + append-only `manager_state_observations`);
* it updates the acquisition ledger (close outgoing stints, open incoming ones
  with ``source="manual"``) so already-executed transfers are represented once
  and are never charged again;
* it verifies the resulting PlanningContext before committing, and fails (with
  a rollback) rather than leaving a state that silently reverts to stale
  official data.

Examples
--------
Dry run (validate the user-confirmed state, write nothing)::

    python scripts/confirm_manager_state.py --event 4 --ft 0 --bank 7 \\
        --transfer 329:305 --source-label user_confirmed_gw4_rodon_to_davis --dry-run

Commit it::

    python scripts/confirm_manager_state.py --event 4 --ft 0 --bank 7 \\
        --transfer 329:305 --source-label user_confirmed_gw4_rodon_to_davis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import repositories as repo
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context
from fpl_brain.scoring_rules import POSITION_IDS
from fpl_brain.utils import utc_now

DEFAULT_SOURCE = "user_confirmed_override"


def _parse_transfer(value: str) -> tuple[int, int]:
    try:
        out_text, in_text = value.split(":", 1)
        return int(out_text), int(in_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(f"transfer must be OUT_ID:IN_ID, got {value!r}") from exc


def _player_meta(conn, player_ids):
    meta = {}
    for row in conn.execute(
        "SELECT id, web_name, full_name, element_type, team_id FROM players WHERE id IN (%s)"
        % ",".join("?" for _ in player_ids),
        [int(pid) for pid in player_ids],
    ):
        record = dict(row)
        meta[int(record["id"])] = {
            "name": record.get("full_name") or record.get("web_name") or str(record["id"]),
            "position": POSITION_IDS.get(int(record["element_type"])) if record.get("element_type") is not None else None,
            "team_id": record.get("team_id"),
        }
    return meta


def _market_price(conn, player_id: int) -> int | None:
    row = repo.latest_snapshot(conn, int(player_id))
    if row is None or row.get("now_cost") is None:
        return None
    return int(row["now_cost"])


def _context_snapshot(conn, entry_id: int, event: int, as_of: str) -> dict:
    context = get_planning_context(conn, entry_id, event, as_of=as_of)
    squad_ids = sorted(int(row["player_id"]) for row in repo.active_manager_acquisitions(conn, entry_id))
    return {
        "context": context,
        "squad_ids": squad_ids,
        "player_count": len(squad_ids),
        "free_transfers": (context.manager_state or {}).get("free_transfers"),
        "bank": (context.manager_state or {}).get("bank"),
        "manager_state": context.manager_state or {},
        "health": context.health,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record a user-confirmed manager state (explicit override)")
    parser.add_argument("--config")
    parser.add_argument("--entry", type=int, help="override configured Team ID")
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--ft", type=int, required=True, help="free transfers REMAINING for the event")
    parser.add_argument("--bank", type=int, required=True, help="bank in integer tenths of £m")
    parser.add_argument("--event-start-ft", type=int,
                        help="free transfers held at the START of the Gameweek (before any transfer). "
                             "Required for a Wildcard/Free-Hit FT transition; never inferred.")
    parser.add_argument("--transfer", action="append", type=_parse_transfer, default=[],
                        metavar="OUT_ID:IN_ID", help="user-confirmed transfer (repeatable)")
    parser.add_argument("--source-label", default=DEFAULT_SOURCE,
                        help="provenance label stored with the override")
    parser.add_argument("--captured-at", help="UTC timestamp for the observation (default: now)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print provenance, write nothing")
    parser.add_argument("--no-strict-bank", action="store_true",
                        help="do not fail when the declared bank disagrees with the ledger arithmetic")
    parser.add_argument("--provenance-out", help="optional path to write the provenance JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.ft < 0:
        print("free transfers must not be negative", file=sys.stderr)
        return 2
    if args.bank < 0:
        print("bank must not be negative", file=sys.stderr)
        return 2
    if args.event_start_ft is not None and args.event_start_ft < 0:
        print("event-start free transfers must not be negative", file=sys.stderr)
        return 2

    config = load_config(args.config)
    entry_id = int(args.entry if args.entry is not None else config["fpl_entry_id"])
    event = int(args.event)
    as_of = args.captured_at or utc_now()
    conn = connect_database(config_path(config, "database"))
    try:
        before = _context_snapshot(conn, entry_id, event, as_of)
        active = {int(row["player_id"]): dict(row) for row in repo.active_manager_acquisitions(conn, entry_id)}
        transfer_ids = [pid for pair in args.transfer for pid in pair]
        meta = _player_meta(conn, sorted(set(transfer_ids) | set(active)))

        # --- validation ----------------------------------------------------------
        failures: list[str] = []
        outgoing: list[int] = []
        incoming: list[int] = []
        for out_id, in_id in args.transfer:
            if out_id == in_id:
                failures.append(f"SAME_PLAYER_OUT_AND_IN: {out_id}")
                continue
            if out_id not in active:
                failures.append(f"OUTGOING_NOT_ACTIVE_IN_LEDGER: {out_id}")
            elif out_id in outgoing:
                failures.append(f"DUPLICATE_OUTGOING: {out_id}")
            if in_id in active:
                failures.append(f"INCOMING_ALREADY_OWNED: {in_id}")
            elif in_id in incoming:
                failures.append(f"DUPLICATE_INCOMING: {in_id}")
            if in_id not in meta:
                failures.append(f"UNKNOWN_PLAYER: {in_id}")
            outgoing.append(out_id)
            incoming.append(in_id)
        for out_id, in_id in zip(outgoing, incoming):
            out_pos = (meta.get(out_id) or {}).get("position")
            in_pos = (meta.get(in_id) or {}).get("position")
            if out_pos and in_pos and out_pos != in_pos:
                failures.append(f"POSITION_MISMATCH: {out_id}({out_pos}) -> {in_id}({in_pos})")

        if args.event_start_ft is not None and args.event_start_ft < int(args.ft):
            failures.append(
                f"EVENT_START_FT_BELOW_REMAINING: start={int(args.event_start_ft)} remaining={int(args.ft)}"
            )

        # Bank arithmetic cross-check (Phase-7A selling-price rule).
        predicted_bank = int(before["bank"] or 0)
        proceeds = 0
        cost = 0
        for out_id, in_id in zip(outgoing, incoming):
            if out_id not in active:
                continue  # already reported as OUTGOING_NOT_ACTIVE_IN_LEDGER
            market = _market_price(conn, out_id)
            purchase = int(active[out_id]["purchase_price"])
            if market is None:
                failures.append(f"MISSING_MARKET_PRICE: {out_id}")
                continue
            proceeds += market if market <= purchase else purchase + (market - purchase) // 2
            incoming_price = _market_price(conn, in_id)
            if incoming_price is None:
                failures.append(f"MISSING_MARKET_PRICE: {in_id}")
                continue
            cost += incoming_price
        predicted_bank = predicted_bank + proceeds - cost
        bank_matches = predicted_bank == int(args.bank)
        if not bank_matches and not args.no_strict_bank:
            failures.append(
                f"BANK_DOES_NOT_RECONCILE: declared={int(args.bank)} ledger_arithmetic={predicted_bank} "
                f"(proceeds={proceeds}, cost={cost})"
            )

        # Resulting squad legitimacy (positions, club limit, size).
        resulting = {pid: row for pid, row in active.items() if pid not in set(outgoing)}
        for in_id in [pid for pid in incoming if pid not in active]:
            market = _market_price(conn, in_id) or 0
            resulting[in_id] = {"player_id": in_id, "purchase_price": market}
        if len(resulting) != 15:
            failures.append(f"RESULTING_SQUAD_NOT_15: {len(resulting)}")
        position_counts: dict[str, int] = {}
        club_counts: dict[int, int] = {}
        for pid in resulting:
            info = meta.get(pid) or {}
            position = info.get("position")
            if position:
                position_counts[position] = position_counts.get(position, 0) + 1
            team = info.get("team_id")
            if team is not None:
                club_counts[int(team)] = club_counts.get(int(team), 0) + 1
        expected_composition = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        for position, required in expected_composition.items():
            if position_counts.get(position, 0) != required:
                failures.append(f"POSITION_COMPOSITION: {position}={position_counts.get(position, 0)} != {required}")
        over_limit = {team: count for team, count in club_counts.items() if count > 3}
        if over_limit:
            failures.append(f"CLUB_LIMIT_EXCEEDED: {over_limit}")

        provenance = {
            "operation": "confirm_manager_state",
            "entry_id": entry_id,
            "event": event,
            "captured_at": as_of,
            "source_label": args.source_label,
            "dry_run": bool(args.dry_run),
            "official_state": {
                "authoritative_source_before": before["manager_state"].get("authoritative_source"),
                "free_transfers": before["free_transfers"],
                "bank": before["bank"],
                "api": before["manager_state"].get("official_api"),
                "manager_sync_run": (before["context"].official_runs or {}).get("manager_sync"),
                "official_runs_fetch": (before["context"].official_runs or {}).get("fetch"),
            },
            "user_confirmed_override": {
                "free_transfers_remaining": int(args.ft),
                "event_start_free_transfers": (None if args.event_start_ft is None else int(args.event_start_ft)),
                "bank_tenths": int(args.bank),
                "transfers": [{"out": o, "in": i} for o, i in zip(outgoing, incoming)],
                "source": args.source_label,
                "captured_at": as_of,
            },
            "field_provenance_after": {
                "free_transfers": "manual (user-confirmed override)",
                "event_start_free_transfers": (
                    "manual (user-confirmed override)"
                    if args.event_start_ft is not None else "data_gap (not stated)"
                ),
                "bank": "manual (user-confirmed override)",
                "ownership": "acquisition ledger (manual source for confirmed transfers)",
            },
            "ledger_effect": {
                "closed": outgoing,
                "inserted": incoming,
                "already_executed_not_recharged": True,
            },
            "bank_arithmetic": {
                "bank_before_tenths": int(before["bank"] or 0),
                "sale_proceeds_tenths": proceeds,
                "purchase_cost_tenths": cost,
                "computed_bank_tenths": predicted_bank,
                "declared_bank_tenths": int(args.bank),
                "matches": bool(bank_matches),
            },
            "ft_arithmetic": {
                "free_transfers_remaining": int(args.ft),
                "event_start_free_transfers": (None if args.event_start_ft is None else int(args.event_start_ft)),
                "event_start_ft_note": (
                    "explicit provenance; a Wildcard/Free Hit played after these transfers must "
                    "preserve THIS bank, not the remaining count"
                ),
                "confirmed_transfers_already_executed": len(incoming),
                "confirmed_transfers_charged_again": 0,
                "already_executed_transfers_recharged": False,
                "next_additional_transfer_hit_points": (0 if int(args.ft) > 0 else 4),
                # Derived, never hardcoded: the event number and the free-transfer
                # count come from the arguments actually being confirmed.
                "note": (
                    "the confirmed transfers were already executed and are represented in the starting "
                    "squad/bank/FT; they are never charged again. "
                    + (
                        f"While free transfers remaining is {int(args.ft)}, any ADDITIONAL GW{int(args.event)} "
                        f"transfer is a paid transfer (-{4 if int(args.ft) <= 0 else 4}) unless a Wildcard is "
                        "activated."
                        if int(args.ft) <= 0
                        else f"{int(args.ft)} free transfer(s) remain for GW{int(args.event)}; an additional "
                        f"transfer beyond those, or the second one while only 1 remains, costs 4 points "
                        "unless a Wildcard is activated."
                    )
                ),
            },
            "validation": {"failures": failures, "passed": not failures},
        }

        if failures:
            provenance["status"] = "FAILED_VALIDATION"
            _emit(provenance, args)
            print("confirm_manager_state FAILED validation; nothing written:", file=sys.stderr)
            for reason in failures:
                print(f"  - {reason}", file=sys.stderr)
            return 3

        if args.dry_run:
            provenance["status"] = "DRY_RUN_OK"
            _emit(provenance, args)
            print(f"dry run OK: entry={entry_id} GW{event} ft={args.ft} bank={args.bank} "
                  f"transfers={len(incoming)} (nothing written)")
            return 0

        # --- apply atomically, then verify before keeping it ---------------------
        try:
            with conn:
                for out_id in outgoing:
                    repo.close_manager_acquisition(
                        conn, int(active[out_id]["id"]), event, sold_at=as_of, updated_at=as_of
                    )
                for in_id in incoming:
                    repo.insert_manager_acquisition(
                        conn, entry_id, in_id, event, int(_market_price(conn, in_id) or 0),
                        source="manual", acquired_at=as_of,
                    )
                repo.upsert_manual_manager_state(
                    conn, entry_id, event, int(args.ft), int(args.bank),
                    source=args.source_label, captured_at=as_of,
                    event_start_free_transfers=(
                        None if args.event_start_ft is None else int(args.event_start_ft)
                    ),
                )
                after = _context_snapshot(conn, entry_id, event, as_of)
                verification = _verify(after, before, outgoing, incoming, args.ft, args.bank,
                                       args.event_start_ft)
                if not verification["passed"]:
                    raise RuntimeError("verification failed: " + "; ".join(verification["failures"]))
        except (RuntimeError, ValueError) as exc:
            print(f"confirm_manager_state rolled back: {exc}", file=sys.stderr)
            return 4

        provenance["status"] = "COMMITTED"
        provenance["verification"] = verification
        provenance["after"] = {
            "squad_ids": after["squad_ids"],
            "player_count": after["player_count"],
            "free_transfers": after["free_transfers"],
            "event_start_free_transfers": after["manager_state"].get("event_start_free_transfers"),
            "bank": after["bank"],
            "authoritative_source": after["manager_state"].get("authoritative_source"),
            "field_provenance": after["manager_state"].get("field_provenance"),
            "stale_official_state_overridden": after["manager_state"].get("stale_official_state_overridden"),
            "health": after["health"].get("status"),
            "health_fail_reasons": after["health"].get("fail_reasons"),
        }
        _emit(provenance, args)
        print(f"confirmed manager state: entry={entry_id} GW{event} ft={args.ft} bank={args.bank} "
              f"squad={after['player_count']} health={after['health'].get('status')}")
        return 0
    finally:
        conn.close()


def _verify(after, before, outgoing, incoming, ft, bank, event_start_ft=None) -> dict:
    failures: list[str] = []
    if after["player_count"] != 15:
        failures.append(f"squad is {after['player_count']}, expected 15")
    if after["free_transfers"] != int(ft):
        failures.append(f"free transfers resolved to {after['free_transfers']}, expected {int(ft)}")
    if after["bank"] != int(bank):
        failures.append(f"bank resolved to {after['bank']}, expected {int(bank)}")
    owned = set(after["squad_ids"])
    for out_id in outgoing:
        if out_id in owned:
            failures.append(f"outgoing player {out_id} still owned")
    for in_id in incoming:
        if in_id not in owned:
            failures.append(f"incoming player {in_id} not owned")
    if after["manager_state"].get("override_authoritative") is not True:
        failures.append("manual override is not authoritative after write")
    for field in ("free_transfers", "bank"):
        if after["manager_state"].get(f"{field}_source") != "manual":
            failures.append(f"{field} did not resolve from the manual override")
    if event_start_ft is not None:
        resolved_start = after["manager_state"].get("event_start_free_transfers")
        if resolved_start is None or int(resolved_start) != int(event_start_ft):
            failures.append(
                f"event_start_free_transfers resolved to {resolved_start}, expected {int(event_start_ft)}"
            )
    if after["health"].get("status") == "FAIL":
        failures.append("PlanningContext health FAIL: " + "; ".join(after["health"].get("fail_reasons") or []))
    return {"passed": not failures, "failures": failures, "before_player_count": before["player_count"]}


def _emit(provenance: dict, args) -> None:
    if args.provenance_out:
        Path(args.provenance_out).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
