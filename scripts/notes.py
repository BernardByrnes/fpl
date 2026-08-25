#!/usr/bin/env python3
"""Small CLI for human watchlist, decision, and strategy records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import repositories as repo
from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.utils import configure_logging


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain FPL Brain human context")
    _add_common(parser)
    commands = parser.add_subparsers(dest="domain", required=True)

    watchlist = commands.add_parser("watchlist")
    watch_commands = watchlist.add_subparsers(dest="action", required=True)
    add_watch = watch_commands.add_parser("add")
    add_watch.add_argument("--player-id", type=int, required=True)
    add_watch.add_argument("--status", choices=["WATCH", "BUY", "HOLD", "AVOID", "SELL"], default="WATCH")
    add_watch.add_argument("--reason")
    add_watch.add_argument("--from-event", type=int)
    add_watch.add_argument("--to-event", type=int)
    add_watch.add_argument("--notes")
    list_watch = watch_commands.add_parser("list")
    update_watch = watch_commands.add_parser("update")
    update_watch.add_argument("id", type=int)
    update_watch.add_argument("--status", choices=["WATCH", "BUY", "HOLD", "AVOID", "SELL"])
    update_watch.add_argument("--reason")
    update_watch.add_argument("--notes")
    update_watch.add_argument("--inactive", action="store_true")

    decision = commands.add_parser("decision")
    decision_commands = decision.add_subparsers(dest="action", required=True)
    add_decision = decision_commands.add_parser("add")
    add_decision.add_argument("--event", type=int, required=True)
    add_decision.add_argument("--action", choices=["TRANSFER", "HOLD", "CAPTAIN", "CHIP", "BENCH", "ROLL_TRANSFER"], required=True)
    add_decision.add_argument("--captain-id", type=int)
    add_decision.add_argument("--vice-captain-id", type=int)
    add_decision.add_argument("--player-in-id", type=int)
    add_decision.add_argument("--player-out-id", type=int)
    add_decision.add_argument("--chip")
    add_decision.add_argument("--reasoning")
    add_decision.add_argument("--confidence", choices=["low", "medium", "high"])
    add_decision.add_argument("--assumptions", default="[]")
    add_decision.add_argument("--invalidators", default="[]")
    add_decision.add_argument("--expected-cost", type=int)
    decision_commands.add_parser("list")
    review = decision_commands.add_parser("review")
    review.add_argument("id", type=int)
    review.add_argument("--notes", required=True)

    strategy = commands.add_parser("strategy")
    strategy_commands = strategy.add_subparsers(dest="action", required=True)
    set_strategy = strategy_commands.add_parser("set")
    set_strategy.add_argument("--event", type=int)
    set_strategy.add_argument("--risk-posture")
    set_strategy.add_argument("--wildcard-horizon")
    set_strategy.add_argument("--bench-boost-plan")
    set_strategy.add_argument("--free-hit-plan")
    set_strategy.add_argument("--triple-captain-plan")
    set_strategy.add_argument("--notes")
    strategy_commands.add_parser("show")

    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        configure_logging(config_path(config, "raw_dir"), args.verbose, args.quiet)
        conn = connect_database(config_path(config, "database"))
        try:
            if args.domain == "watchlist" and args.action == "add":
                with conn:
                    ident = repo.add_watchlist(conn, args.player_id, args.status, args.reason, args.from_event, args.to_event, args.notes)
                print(f"watchlist added id={ident}")
            elif args.domain == "watchlist" and args.action == "update":
                values = {"status": args.status, "reason": args.reason, "notes": args.notes}
                if args.inactive:
                    values["is_active"] = 0
                with conn:
                    repo.update_watchlist(conn, args.id, **{key: value for key, value in values.items() if value is not None})
                print(f"watchlist updated id={args.id}")
            elif args.domain == "watchlist":
                print(json.dumps(repo.active_watchlist_rows(conn), indent=2, ensure_ascii=False))
            elif args.domain == "decision" and args.action == "add":
                assumptions = json.loads(args.assumptions)
                invalidators = json.loads(args.invalidators)
                if not isinstance(assumptions, list) or not isinstance(invalidators, list):
                    raise ValueError("assumptions and invalidators must be JSON arrays")
                with conn:
                    ident = repo.add_decision(conn, vars(args) | {"assumptions": assumptions, "invalidators": invalidators})
                print(f"decision added id={ident}")
            elif args.domain == "decision" and args.action == "review":
                with conn:
                    repo.review_decision(conn, args.id, args.notes)
                print(f"decision reviewed id={args.id}")
            elif args.domain == "decision":
                print(json.dumps(repo.decision_rows(conn), indent=2, ensure_ascii=False))
            elif args.domain == "strategy" and args.action == "set":
                with conn:
                    ident = repo.add_strategy(conn, vars(args))
                print(f"strategy update added id={ident}")
            else:
                print(json.dumps(repo.strategy_row(conn), indent=2, ensure_ascii=False))
        finally:
            conn.close()
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        print(f"notes command failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

