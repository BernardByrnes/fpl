#!/usr/bin/env python3
"""CLI wrapper for report rendering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.reports import build_report, render_json, render_markdown
from fpl_brain.utils import configure_logging, ensure_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the FPL Brain evidence pack")
    parser.add_argument("--config")
    parser.add_argument("--gw", type=int)
    parser.add_argument("--format", choices=["md", "json", "both"], default="both")
    parser.add_argument("--out")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        configure_logging(config_path(config, "raw_dir"), args.verbose, args.quiet)
        conn = connect_database(config_path(config, "database"))
        try:
            report = build_report(conn, config, args.gw)
        finally:
            conn.close()
        export_dir = ensure_directory(config_path(config, "exports_dir"))
        gw = report["title"].split("GW", 1)[1]
        if args.out:
            output = Path(args.out)
            if not output.is_absolute():
                output = Path.cwd() / output
        else:
            output = export_dir / f"gw{gw}_report.md"
        written: list[Path] = []
        if args.format in {"md", "both"}:
            md_path = output if output.suffix.lower() in {".md", ".markdown"} else output.with_suffix(".md")
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(render_markdown(report), encoding="utf-8")
            written.append(md_path)
        if args.format in {"json", "both"}:
            json_path = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(render_json(report), encoding="utf-8")
            written.append(json_path)
        print("wrote " + ", ".join(str(path) for path in written))
        return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"report failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

