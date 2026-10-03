"""Explicit operator CLI for prospective BB/TC causal evidence collection."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import database
from fpl_brain.chip_evidence_ops import (
    ChipEvidenceError,
    capture_outcome,
    forecast_origin,
    mature_observation,
    register_origin,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="existing FPL Brain SQLite database")
    parser.add_argument("--entry-id", required=True, type=int)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--planning-event", required=True, type=int)
    parser.add_argument("--origin-cutoff", required=True)
    parser.add_argument("--evidence-root", required=True)


def _identity_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "db_path": args.db,
        "entry_id": args.entry_id,
        "generation_id": args.generation_id,
        "decision_id": args.decision_id,
        "route_id": args.route_id,
        "planning_event": args.planning_event,
        "cutoff": args.origin_cutoff,
        "evidence_root": args.evidence_root,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chip_evidence.py")
    commands = parser.add_subparsers(dest="command", required=True)

    origin = commands.add_parser("register-origin", help="verify and immutably register one normal origin")
    _common(origin)

    forecast = commands.add_parser("forecast", help="retain one prospective BB or TC forecast")
    _common(forecast)
    forecast.add_argument("--action", choices=("BB", "TC"), required=True)
    forecast.add_argument("--expiry-event", required=True, type=int)
    forecast.add_argument("--observation-id", required=True)
    forecast.add_argument("--continuation-generation-id")
    forecast.add_argument("--cache-dir", required=True)
    forecast.add_argument(
        "--materialize-worlds", action="store_true",
        help="explicitly permit missing matrices only in this origin's isolated cache directory",
    )

    capture = commands.add_parser("capture-outcome", help="append final official event-grain outcome rows")
    _common(capture)
    capture.add_argument("--action", choices=("BB", "TC"), required=True)
    capture.add_argument("--observation-id", required=True)
    capture.add_argument("--realization-event", required=True, type=int)
    capture.add_argument("--fetch-run-id", required=True, type=int)
    capture.add_argument("--raw-root", required=True)

    mature = commands.add_parser("mature", help="finalize one selected event and retain its receipt")
    _common(mature)
    mature.add_argument("--action", choices=("BB", "TC"), required=True)
    mature.add_argument("--observation-id", required=True)
    mature.add_argument("--realization-event", required=True, type=int)
    return parser


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    connection: sqlite3.Connection | None = None
    try:
        if args.command == "capture-outcome":
            database_path = Path(args.db).expanduser().resolve()
            if not database_path.is_file():
                raise ChipEvidenceError(f"database does not exist: {database_path}")
            connection = sqlite3.connect(str(database_path), timeout=30.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=30000")
            result = capture_outcome(
                connection,
                **_identity_args(args),
                action=args.action,
                observation_id=args.observation_id,
                realization_event=args.realization_event,
                fetch_run_id=args.fetch_run_id,
                raw_root=args.raw_root,
            )
        else:
            connection = database.connect_readonly_database(args.db)
            if args.command == "register-origin":
                result = register_origin(connection, **_identity_args(args))
            elif args.command == "forecast":
                result = forecast_origin(
                    connection,
                    **_identity_args(args),
                    action=args.action,
                    expiry_event=args.expiry_event,
                    observation_id=args.observation_id,
                    continuation_generation_id=args.continuation_generation_id,
                    cache_dir=args.cache_dir,
                    materialize_worlds=bool(args.materialize_worlds),
                )
            elif args.command == "mature":
                result = mature_observation(
                    connection,
                    **_identity_args(args),
                    action=args.action,
                    observation_id=args.observation_id,
                    realization_event=args.realization_event,
                )
            else:  # pragma: no cover - argparse enforces the closed command set
                raise ChipEvidenceError(f"unsupported command {args.command}")
        _emit(result)
        return 0
    except (ChipEvidenceError, sqlite3.Error, OSError, ValueError) as failure:
        print(f"chip evidence refused: {failure}", file=sys.stderr)
        return 2
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
