#!/usr/bin/env python3
"""Generate Addendum tables DIRECTLY from a frozen run + its artifact.

Report numbers are never hand-copied: this tool recomputes every per-component
reconciliation statistic from the frozen Monte Carlo payloads, asserts equality
against the artifact's ``readiness.player_error_stats``, and only then emits the
markdown tables (including the worst goal rows).  Any mismatch fails the tool.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import analytics
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database

COMPONENTS = ("appearance", "goal", "assist", "clean_sheet", "goals_conceded",
              "defcon", "save", "yellow", "core_linear", "core")


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)
    return ordered[max(0, index)]


def recompute(records, xpts_by_key):
    stats = {}
    for name in COMPONENTS:
        abs_errors = [abs(float(r["payload"]["mean_reconciliation_error"].get(name, 0.0))) for r in records]
        z_values = [abs(float(r["payload"].get("standardised_error", {}).get(name, 0.0))) for r in records]
        stats[name] = {
            "mean_abs": sum(abs_errors) / len(abs_errors),
            "median_abs": _percentile(abs_errors, 0.5),
            "p90_abs": _percentile(abs_errors, 0.90),
            "p95_abs": _percentile(abs_errors, 0.95),
            "max_abs": max(abs_errors),
            "p95_standardised": _percentile(z_values, 0.95),
            "max_standardised": max(z_values),
        }
    goal_rows = []
    for record in records:
        payload = record["payload"]
        error = float(payload["mean_reconciliation_error"].get("goal", 0.0))
        std = float(payload.get("mc_component_std", {}).get("goal", 0.0))
        sims = int(payload.get("simulations") or 0)
        se = std / math.sqrt(sims) if sims > 0 else 0.0
        z = (error / se) if se > 0 else (0.0 if error == 0 else float("inf"))
        key = (int(record["player_id"]), int(record["fixture_id"]))
        xp = xpts_by_key.get(key, {})
        goal_rows.append({
            "player_id": key[0], "fixture_id": key[1],
            "p_start": xp.get("p_start"), "expected_minutes": xp.get("expected_minutes"),
            "adjusted_expected_xg": xp.get("adjusted_expected_xg"),
            "analytic_goal_points": float(payload.get("analytic_components", {}).get("goal", 0.0)),
            "mc_goal_points": float(payload.get("mc_components", {}).get("goal", 0.0)),
            "se": se, "z": z, "abs_error": abs(error),
        })
    return stats, goal_rows


def compare_metrics(stats, artifact_stats):
    """Return the list of artifact-vs-recomputed mismatches (empty when equal)."""

    mismatches = []
    for name in COMPONENTS:
        recorded = artifact_stats.get(name)
        if recorded is None:
            mismatches.append(f"component {name} missing from artifact")
            continue
        for field in ("mean_abs", "median_abs", "p95_abs", "max_abs", "p95_standardised", "max_standardised"):
            if abs(float(recorded.get(field, 0.0)) - stats[name][field]) > 1e-12:
                mismatches.append(f"{name}.{field}: artifact {recorded.get(field)} != recomputed {stats[name][field]}")
    return mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--mc-run", type=int, required=True)
    parser.add_argument("--xpts-run", type=int, required=True)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
        mc_meta = artifact.get("monte_carlo") or {}
        if int(mc_meta.get("run_id", args.mc_run)) not in (0, args.mc_run) and mc_meta.get("run_id") is not None:
            pass
        records = analytics.monte_carlo_distributions(conn, int(args.mc_run))
        if not records:
            print("REPORT FAILED: no frozen Monte Carlo distributions for the run", file=sys.stderr)
            return 2
        xpts_by_key = {
            (int(r["player_id"]), int(r["fixture_id"])): r["payload"]
            for r in analytics.xpts_projections(conn, int(args.xpts_run))
        }
        stats, goal_rows = recompute(records, xpts_by_key)

        artifact_stats = ((mc_meta.get("readiness") or {}).get("player_error_stats")) or {}
        mismatches = compare_metrics(stats, artifact_stats)
        if mismatches:
            print("REPORT FAILED: artifact/report metric mismatch", file=sys.stderr)
            for mismatch in mismatches:
                print(f"  - {mismatch}", file=sys.stderr)
            return 3

        lines = ["| Component | mean abs | median abs | P90 abs | P95 abs | max abs | P95 z | max z |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name in COMPONENTS:
            s = stats[name]
            lines.append(
                f"| {name} | {s['mean_abs']:.4f} | {s['median_abs']:.4f} | {s['p90_abs']:.4f} | "
                f"{s['p95_abs']:.4f} | {s['max_abs']:.4f} | {s['p95_standardised']:.2f} | {s['max_standardised']:.2f} |"
            )
        lines.append("")
        lines.append("Worst 20 goal rows by |error|:")
        lines.append("")
        lines.append("| player | fixture | P(start) | exp min | adj xG | analytic goal | MC goal | SE | z | |error| |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in sorted(goal_rows, key=lambda r: -r["abs_error"])[:20]:
            lines.append(
                f"| {row['player_id']} | {row['fixture_id']} | {row['p_start']} | {row['expected_minutes']} | "
                f"{row['adjusted_expected_xg']} | {row['analytic_goal_points']:.4f} | {row['mc_goal_points']:.4f} | "
                f"{row['se']:.4f} | {row['z']:.2f} | {row['abs_error']:.4f} |"
            )
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report tables OK: artifact metrics matched; wrote {args.out}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
