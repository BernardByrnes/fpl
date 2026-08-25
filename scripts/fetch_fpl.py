#!/usr/bin/env python3
"""CLI wrapper for the official-data fetch."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.api import FplApiError
from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.ingest import run_fetch
from fpl_brain.utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public FPL facts into the local database")
    parser.add_argument("--config", help="path to config JSON")
    parser.add_argument("--summaries", choices=["none", "squad", "all"], default="none")
    parser.add_argument("--live", action="store_true", help="also fetch event live data")
    parser.add_argument("--gw", type=int, help="gameweek for --live")
    parser.add_argument("--dry-run", action="store_true", help="fetch and parse without writing")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if not args.dry_run:
            configure_logging(config_path(config, "raw_dir"), args.verbose, args.quiet)
        else:
            logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
        result = run_fetch(config, args.summaries, args.gw if args.live else None, args.dry_run)
        counts = result.get("counts") or {
            "teams": result.get("teams", 0),
            "players": result.get("players", 0),
            "fixtures": result.get("fixtures", 0),
            "snapshots": result.get("snapshots", 0),
        }
        print(
            f"fetch status={result['status']} teams={counts.get('teams', 0)} players={counts.get('players', 0)} "
            f"fixtures={counts.get('fixtures', 0)} snapshots={counts.get('snapshots', 0)}"
        )
        return 1 if result["status"] == "partial" else 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (FplApiError, OSError, ValueError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

