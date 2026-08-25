#!/usr/bin/env python3
"""CLI wrapper for public manager synchronization."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.api import FplApiError
from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.ingest import run_manager_sync
from fpl_brain.utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync one public FPL manager")
    parser.add_argument("--config", help="path to config JSON")
    parser.add_argument("--entry", type=int, help="override configured Team ID")
    parser.add_argument("--event", type=int)
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.entry is not None:
            config["fpl_entry_id"] = args.entry
        if not args.dry_run:
            configure_logging(config_path(config, "raw_dir"), args.verbose, args.quiet)
        else:
            logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
        result = run_manager_sync(config, args.event, args.all_events, args.dry_run)
        print(result.get("message") or f"manager sync status={result['status']} entry={result.get('entry_id', '—')} event={result.get('event', '—')} picks={result.get('picks_written', 0)}")
        return 1 if result.get("status") == "partial" else 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (FplApiError, OSError, ValueError) as exc:
        print(f"manager sync failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

