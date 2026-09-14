#!/usr/bin/env python3
"""REPAIR A: refresh required completed-event history before certifying.

Run this between the official source refresh and predictive certification.  It

  1. derives the planning event and the cutoff,
  2. audits completed-event history for that cutoff,
  3. if anything is missing, refreshes it through the canonical official ingest
     (``ingest.run_fetch`` with ``summaries="all"``),
  4. re-audits, and
  5. exits NON-ZERO when the required history is still incomplete.

Why ``summaries="all"`` and not the per-event live endpoint: the live endpoint
carries minutes and points but no xG, and ``player_rates`` fits xG/90 and xA/90.
The element-summary payload is the only official source that carries the xG
family, so it is the smallest refresh that satisfies the models' requirements.

The refresh is a NO-OP when history is already complete, and it NEVER invents
zeros: when official history cannot be obtained the audit still fails and the
process exits non-zero so no certification can follow.

Usage:
    python scripts/refresh_completed_history.py --config config.json
        [--planning-event 5] [--cutoff <UTC>] [--dry-run] [--out <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import history_completeness as hc
from fpl_brain.config import ConfigError, load_config
from fpl_brain.utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh completed-event player history for certification")
    parser.add_argument("--config", help="path to config JSON")
    parser.add_argument("--planning-event", type=int, help="event being planned for (defaults to the official next event)")
    parser.add_argument("--cutoff", help="causal cutoff UTC (defaults to the freshest official capture)")
    parser.add_argument("--dry-run", action="store_true", help="audit and report without ingesting")
    parser.add_argument("--out", help="optional path to write the JSON report")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        configure_logging(Path(config["paths"]["raw_dir"]), args.verbose, args.quiet)
        report = hc.refresh_completed_event_history(
            config,
            planning_event=args.planning_event,
            cutoff=args.cutoff,
            dry_run=args.dry_run,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except hc.HistoryRefreshIncomplete as exc:
        print(f"history completeness REFUSED: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"history refresh failed: {exc}", file=sys.stderr)
        return 1

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
    after = report["after"]
    print(
        f"history completeness: complete={report['complete']} "
        f"planning_event={report['planning_event']} cutoff={report['cutoff']} "
        f"required_completed_events={after['required_completed_events']} "
        f"latest_required={after['latest_required_completed_event']} "
        f"in_progress_events={after['in_progress_events']} "
        f"refreshed={report['refresh']['performed']}"
    )
    if not report["complete"]:
        print(
            f"{hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE}: {after['reasons']}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
