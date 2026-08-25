#!/usr/bin/env python3
"""Validate a scouting JSON file without writing anything (dry-run wrapper).

Reuses the implemented importer's validation and resolution logic; no
duplicate validator is maintained here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, load_config
from fpl_brain.scouting import DuplicateScoutingImport, ScoutingError, import_scouting


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate scouting JSON against the importer (no DB writes)")
    parser.add_argument("source")
    parser.add_argument("--config")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        try:
            result = import_scouting(config, args.source, dry_run=True)
        except DuplicateScoutingImport as exc:
            print(f"VALID-DUPLICATE: {exc}")
            print("The file content was already imported; importing again requires scripts/import_scouting.py --force.")
            return 0
        print(
            f"VALID: parse OK, root OK, observations validated, players resolved "
            f"{result['players_resolved']}/{result['players_total']}, notes {result['notes']}, "
            f"no database writes performed"
        )
        if result["unresolved"]:
            print("Unresolved players (would be rejected to scouting/rejected/ on import):")
            for player in result["unresolved"]:
                print(f"  - {player.get('player_name')}: {player.get('reason')}")
                for candidate in player.get("candidates", [])[:5]:
                    print(
                        f"      candidate id={candidate.get('id')} full={candidate.get('full_name')} "
                        f"web={candidate.get('web_name')} team={candidate.get('team')}"
                    )
            print("Review unresolved players before importing; resolved players would still import.")
            return 1
        return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except ScoutingError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
