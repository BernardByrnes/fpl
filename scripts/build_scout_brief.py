#!/usr/bin/env python3
"""Generate the weekly scout research brief from the FPL Brain database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.scout_ops import load_scout_plan, render_brief_markdown
from fpl_brain.scout_ops import build_brief
from fpl_brain.utils import ensure_directory
from fpl_brain import repositories as repo


def _timestamp(now: str) -> str:
    return now.replace("-", "").replace(":", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Luna scout research brief")
    parser.add_argument("--gw", type=int, help="gameweek (default: current or next event)")
    parser.add_argument("--config")
    parser.add_argument("--plan", default="config/scout_plan.json", help="manual weekly plan file")
    parser.add_argument("--out-dir", default="data/scouting/briefs")
    parser.add_argument("--format", choices=["md", "json", "both"], default="both")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        database = config_path(config, "database")
        if not Path(database).exists():
            print(f"brief failed: database {database} does not exist — run scripts/fetch_fpl.py first", file=sys.stderr)
            return 1
        plan, plan_warnings = load_scout_plan(args.plan)
        conn = connect_database(database)
        try:
            gw = args.gw or repo.current_or_next_event(conn)
            brief = build_brief(conn, config, gw, plan)
        finally:
            conn.close()
        stamp = _timestamp(brief["generated"])
        base = Path(args.out_dir) / f"gw{gw:02d}_main_{stamp}"
        written: list[Path] = []
        if args.format in {"md", "both"}:
            ensure_directory(base.parent)
            md_path = base.with_suffix(".md")
            md_path.write_text(render_brief_markdown(brief), encoding="utf-8")
            written.append(md_path)
        if args.format in {"json", "both"}:
            ensure_directory(base.parent)
            json_path = base.with_suffix(".json")
            json_path.write_text(json.dumps(brief, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(json_path)
        print("wrote " + ", ".join(str(path) for path in written))
        if plan_warnings:
            for warning in plan_warnings:
                print(f"warning: {warning}", file=sys.stderr)
        if brief["manual_input_gaps"]:
            for gap in brief["manual_input_gaps"]:
                print(f"manual input: {gap}")
        return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"brief failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
