#!/usr/bin/env python3
"""Print the operational scout-coverage diagnostic (no research, no writes)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.scout_ops import load_scout_plan, render_status_text, scout_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scout coverage status for a gameweek")
    parser.add_argument("--gw", type=int, help="gameweek (default: current or next event)")
    parser.add_argument("--config")
    parser.add_argument("--plan", default="config/scout_plan.json", help="manual weekly plan file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        database = config_path(config, "database")
        if not Path(database).exists():
            print(f"status failed: database {database} does not exist — run scripts/fetch_fpl.py first", file=sys.stderr)
            return 1
        plan, _ = load_scout_plan(args.plan)
        conn = connect_database(database)
        try:
            status = scout_status(conn, config, args.gw, plan)
        finally:
            conn.close()
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
        print(render_status_text(status))
        return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"status failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
