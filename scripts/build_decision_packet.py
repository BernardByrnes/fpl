#!/usr/bin/env python3
"""Build the canonical decision packet (JSON + deterministic Markdown)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.packet import (
    build_decision_packet,
    packet_sha256,
    packet_to_json,
    render_packet_markdown,
    verify_packet_markdown,
)
from fpl_brain.utils import utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the canonical FPL Brain decision packet")
    parser.add_argument("--config")
    parser.add_argument("--gw", type=int, help="planning gameweek; defaults to current-or-next official event")
    parser.add_argument("--as-of", help="state lookback timestamp; defaults to build time")
    parser.add_argument("--out-dir", help="output directory; defaults to data/decisions/packets")
    parser.add_argument("--route-out", type=int, action="append", help="transfer-out player ID (repeatable)")
    parser.add_argument("--route-in", type=int, action="append", help="transfer-in player ID (repeatable)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        event = args.gw
        if event is None:
            conn_probe = connect_database(config_path(config, "database"))
            try:
                event = _current_or_next(conn_probe)
            finally:
                conn_probe.close()
        as_of = args.as_of or utc_now()
        conn = connect_database(config_path(config, "database"))
        try:
            built = build_decision_packet(
                conn,
                config,
                event,
                as_of=as_of,
                route_out=args.route_out,
                route_in=args.route_in,
            )
        finally:
            conn.close()
        packet = built["packet"]
        markdown = render_packet_markdown(packet)
        if not verify_packet_markdown(markdown, packet):
            raise ValueError("deterministic renderer disagreed with the canonical packet")
        out_dir = Path(args.out_dir) if args.out_dir else config_path(config, "exports_dir") / "packets"
        out_dir.mkdir(parents=True, exist_ok=True)
        gw = int(packet["metadata"]["event"])
        stem = out_dir / f"decision_packet_gw{gw:02d}"
        json_path = stem.with_suffix(".json")
        md_path = stem.with_suffix(".md")
        json_path.write_text(packet_to_json(packet), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        print(
            f"decision packet written: {json_path} {md_path} "
            f"health={packet['metadata']['health']['status']} packets={packet_sha256(packet)}"
        )
        return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"decision packet failed: {exc}", file=sys.stderr)
        return 1


def _current_or_next(conn) -> int:
    row = conn.execute("SELECT id FROM events WHERE is_current=1 ORDER BY id LIMIT 1").fetchone()
    if row:
        return int(row[0])
    row = conn.execute("SELECT id FROM events WHERE is_next=1 ORDER BY id LIMIT 1").fetchone()
    return int(row[0]) if row else 1


if __name__ == "__main__":
    raise SystemExit(main())
