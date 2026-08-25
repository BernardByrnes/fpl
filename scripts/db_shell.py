#!/usr/bin/env python3
"""Open the configured database in the sqlite3 CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.database import connect_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open fpl.db in sqlite3")
    parser.add_argument("--config")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if shutil.which("sqlite3") is None:
            conn = connect_database(config_path(config, "database"))
            conn.close()
            print("sqlite3 CLI was not found on PATH; the configured database was initialized.")
            return 0
        return subprocess.call(["sqlite3", str(config_path(config, "database"))])
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
