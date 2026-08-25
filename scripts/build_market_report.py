#!/usr/bin/env python3
"""Build the read-only Post-GW Market Report V1."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.market_report import (
    IncompleteGameweekError,
    MarketReportConfigError,
    MarketReportDataError,
    MarketReportError,
    MarketReportOutputError,
    build_market_report,
    load_market_config,
    open_read_only_database,
    render_market_json,
    render_market_markdown,
    write_market_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic local Post-GW Market Report V1")
    parser.add_argument("--gw", type=int, required=True, help="completed target gameweek")
    parser.add_argument("--config", help="existing FPL Brain configuration path")
    parser.add_argument("--market-config", help="optional Market Report configuration override")
    parser.add_argument("--db", help="optional local SQLite database path override")
    parser.add_argument("--out-dir", help="override Market Report output base directory")
    parser.add_argument("--dry-run", action="store_true", help="calculate and validate without writing files")
    parser.add_argument("--stdout", choices=["markdown", "json"], help="print one artifact and write no files")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow only a dry-run stdout preview for an incomplete/unverifiable gameweek",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_incomplete and (not args.dry_run or not args.stdout):
        parser.error("--allow-incomplete requires --dry-run and --stdout")
    try:
        brain_config = load_config(args.config)
        market_config = load_market_config(args.market_config)
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        elif args.quiet:
            logging.basicConfig(level=logging.WARNING)
        else:
            logging.basicConfig(level=logging.INFO)
        database = Path(args.db) if args.db else config_path(brain_config, "database")
        conn = open_read_only_database(database)
        try:
            report = build_market_report(
                conn,
                brain_config,
                market_config,
                args.gw,
                db_path=database,
                allow_incomplete=args.allow_incomplete,
            )
        finally:
            conn.close()
        if args.stdout == "markdown":
            sys.stdout.write(render_market_markdown(report))
            return 0
        if args.stdout == "json":
            sys.stdout.write(render_market_json(report))
            return 0
        if args.dry_run:
            print(
                f"dry-run success: GW{args.gw} status={report['gameweek']['status']} "
                f"market_candidates={report['scout_candidates']['market_triggered_count']}"
            )
            return 0
        written = write_market_report(report, market_config, args.out_dir)
        print("wrote " + ", ".join(str(path) for _, path in sorted(written.items())))
        return 0
    except (ConfigError, MarketReportConfigError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except IncompleteGameweekError as exc:
        print(f"market report incomplete: {exc}", file=sys.stderr)
        return 4
    except MarketReportDataError as exc:
        print(f"market report data error: {exc}", file=sys.stderr)
        return 3
    except MarketReportOutputError as exc:
        print(f"market report output error: {exc}", file=sys.stderr)
        return 5
    except (MarketReportError, OSError, ValueError) as exc:
        print(f"market report failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
