"""Explicit operator CLI for prospective chip causal evidence collection."""

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


def _event_generation(value: str) -> tuple[int, str]:
    event_text, separator, generation_id = str(value).partition("=")
    try:
        event = int(event_text)
    except ValueError as failure:
        raise argparse.ArgumentTypeError("value generation must use EVENT=GENERATION_ID") from failure
    if not separator or event <= 0 or not generation_id.strip():
        raise argparse.ArgumentTypeError("value generation must use EVENT=GENERATION_ID")
    return event, generation_id.strip()


def _event_generation_map(values: list[tuple[int, str]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for event, generation_id in values:
        if event in result:
            raise ChipEvidenceError(f"WC value generation event {event} is repeated")
        result[event] = generation_id
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chip_evidence.py")
    commands = parser.add_subparsers(dest="command", required=True)

    origin = commands.add_parser("register-origin", help="verify and immutably register one normal origin")
    _common(origin)

    horizon_origin = commands.add_parser(
        "register-horizon-origin", help="register explicit FH/WC origin eligibility",
    )
    _common(horizon_origin)
    horizon_origin.add_argument("--action", choices=("FH", "WC"), required=True)

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

    horizon_forecast = commands.add_parser(
        "horizon-forecast", help="retain one prospective FH or WC reservation forecast",
    )
    _common(horizon_forecast)
    horizon_forecast.add_argument("--action", choices=("FH", "WC"), required=True)
    horizon_forecast.add_argument("--expiry-event", required=True, type=int)
    horizon_forecast.add_argument("--observation-id", required=True)
    horizon_forecast.add_argument("--continuation-generation-id", required=True)
    horizon_forecast.add_argument("--cache-dir", required=True)
    horizon_forecast.add_argument(
        "--value-horizon-length", type=int,
        help="required for WC; certified value generation event count (6-10)",
    )
    horizon_forecast.add_argument(
        "--value-generation", type=_event_generation, action="append", default=[], metavar="EVENT=ID",
        help="explicit same-origin WC value generation; repeat once per opportunity event",
    )

    capture = commands.add_parser("capture-outcome", help="append final official event-grain outcome rows")
    _common(capture)
    capture.add_argument("--action", choices=("BB", "TC", "FH", "WC"), required=True)
    capture.add_argument("--observation-id", required=True)
    capture.add_argument("--realization-event", required=True, type=int)
    capture.add_argument("--fetch-run-id", required=True, type=int)
    capture.add_argument("--raw-root", required=True)

    mature = commands.add_parser("mature", help="finalize one selected event and retain its receipt")
    _common(mature)
    mature.add_argument("--action", choices=("BB", "TC", "FH", "WC"), required=True)
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
            elif args.command == "register-horizon-origin":
                from fpl_brain.chip_horizon_evidence_ops import register_horizon_origin

                result = register_horizon_origin(
                    connection, **_identity_args(args), action=args.action,
                )
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
            elif args.command == "horizon-forecast":
                from fpl_brain import chip_decision as cd
                from fpl_brain import chip_reservation_forecast as crf
                from fpl_brain import chip_route_assembly as cra
                from fpl_brain import generation_store as gs
                from fpl_brain import chip_evidence_ops as evidence_ops
                from fpl_brain.chip_horizon_evidence_ops import build_future_action_forecast

                value_generation_ids = _event_generation_map(args.value_generation)
                if args.action == "WC":
                    if args.value_horizon_length is None:
                        raise ChipEvidenceError("WC requires --value-horizon-length")
                    if not 6 <= int(args.value_horizon_length) <= 10:
                        raise ChipEvidenceError("WC value horizon length must be between 6 and 10")
                    expected_value_events = set(range(
                        int(args.planning_event) + 1, int(args.expiry_event) + 1,
                    ))
                    if set(value_generation_ids) != expected_value_events:
                        raise ChipEvidenceError(
                            "WC requires exactly one explicit value generation for every event through expiry"
                        )
                elif args.value_horizon_length is not None or value_generation_ids:
                    raise ChipEvidenceError("FH does not accept WILDCARD_VALUE generation arguments")

                identity = evidence_ops._required_identity(**_identity_args(args))
                generation = gs.load_generation(connection, args.generation_id)
                route = cra.load_verified_normal_route(
                    connection, args.decision_id, route_id=args.route_id,
                )
                source_identity = evidence_ops._source_identity(route, generation)
                coverage = crf.build_reservation_coverage_product(
                    connection,
                    action=(cd.CHIP_ACTION_FH if args.action == "FH" else cd.CHIP_ACTION_WC),
                    source_identity=source_identity,
                    expiry_event=args.expiry_event,
                    product_generation_id=args.continuation_generation_id,
                    wildcard_value_horizon_length=args.value_horizon_length,
                )
                result = build_future_action_forecast(
                    connection,
                    identity=identity,
                    action=args.action,
                    observation_id=args.observation_id,
                    entry_id=args.entry_id,
                    continuation_generation_id=args.continuation_generation_id,
                    value_generation_ids=value_generation_ids,
                    expiry_event=args.expiry_event,
                    coverage_product=coverage,
                    evidence_root=args.evidence_root,
                    cache_dir=args.cache_dir,
                )
                result = {**result, "coverage_product_sha256": coverage["product_sha256"]}
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
