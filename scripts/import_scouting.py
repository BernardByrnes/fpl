#!/usr/bin/env python3
"""CLI wrapper for scouting imports."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.scouting import DuplicateScoutingImport, ScoutingError, import_scouting
from fpl_brain.utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import researched scouting context")
    parser.add_argument("source")
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if not args.dry_run:
            configure_logging(config_path(config, "raw_dir"), args.verbose, args.quiet)
        else:
            logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
        result = import_scouting(config, args.source, args.dry_run, args.force)
        print(
            f"scouting {result['status']}: players={result['players_resolved']}/{result['players_total']} "
            f"notes={result['notes']} unresolved={len(result['unresolved'])}"
        )
        return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except DuplicateScoutingImport as exc:
        print(f"scouting import blocked: {exc}", file=sys.stderr)
        return 1
    except (ScoutingError, OSError, ValueError) as exc:
        print(f"scouting import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

