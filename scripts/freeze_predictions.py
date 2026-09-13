#!/usr/bin/env python3
"""Pre-deadline prediction freeze: baselines, Minutes, Team Strength, Player Rates, xPts.

Builds the canonical PlanningContext, requires acceptable analytics readiness,
creates immutable projection runs, freezes the selected model families, and
writes a deterministic summary artifact.  When ``xpts`` is selected its input
run IDs are explicit: either the component runs created in this same
invocation (one synchronized cutoff) or runs passed with ``--xpts-*-run``.

It never alters manager decisions and makes no transfer/captain/chip
recommendation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import (
    analytics,
    causality,
    execution,
    joint_minutes,
    minutes_coherence,
    minutes_model,
    monte_carlo,
    player_rates,
    scoring_rules,
    substitution_model,
    team_model,
    xpts,
)
from fpl_brain.config import ConfigError, config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context
from fpl_brain.utils import parse_utc, utc_now

FAMILIES = ("baseline", "minutes", "minutes_coherent", "minutes_positional", "minutes_substitution", "minutes_joint",
            "team", "rates", "xpts", "monte_carlo")


class _ReadinessFailure(Exception):
    def __init__(self, name: str, reasons: list[str]) -> None:
        super().__init__(name)
        self.name = name
        self.reasons = reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze analytics projections for a gameweek")
    parser.add_argument("--gw", type=int, required=True, help="planning gameweek to freeze")
    parser.add_argument("--config")
    parser.add_argument("--cutoff", help="explicit UTC ISO-8601 planning cutoff (for a coherent horizon bundle)")
    parser.add_argument("--mc-simulations", type=int, default=10000,
                        help="Monte Carlo draws for this freeze (horizon bundles may use a smaller budget)")
    parser.add_argument("--mc-calibration-states", type=int, default=20000,
                        help="scorer/assist calibration state count (solver budget; not a gate/threshold)")
    parser.add_argument("--mc-calibration-iterations", type=int, default=250,
                        help="scorer/assist calibration max iterations (solver budget; not a gate/threshold)")
    parser.add_argument("--families", default="all",
                        help="comma list: baseline,minutes,minutes_coherent,minutes_positional,minutes_substitution,minutes_joint,team,rates,xpts,monte_carlo (default all)")
    parser.add_argument("--xpts-minutes-run", type=int, help="explicit minutes run id for xPts")
    parser.add_argument("--xpts-team-run", type=int, help="explicit team run id for xPts")
    parser.add_argument("--xpts-team-baseline-run", type=int, help="explicit team baseline run id for xPts")
    parser.add_argument("--xpts-rate-run", type=int, help="explicit player-rate run id for xPts")
    parser.add_argument("--xpts-run", type=int, help="explicit xPts run id (for a standalone Monte Carlo run)")
    parser.add_argument("--out-dir", help="override summary artifact directory")
    parser.add_argument("--dry-run", action="store_true", help="build projections and readiness without writing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        selected = set(FAMILIES) if args.families.strip().lower() == "all" else {
            part.strip() for part in args.families.split(",") if part.strip()
        }
        unknown = selected - set(FAMILIES)
        if unknown:
            print(f"configuration error: unknown families {sorted(unknown)}", file=sys.stderr)
            return 2
        config = load_config(args.config)
        conn = connect_database(config_path(config, "database"))
        try:
            if args.dry_run:
                # A dry run writes nothing, so it neither needs nor takes the
                # writer lease; it must not block a real freeze.
                return _freeze(conn, config, args, selected)
            # Production writes go through the execution guard: one run identity,
            # one SQLite writer, and a hard stop that is the official deadline
            # (or the wall-clock ceiling, whichever comes first).  A duplicate
            # invocation for the same semantic work fails here instead of
            # colliding inside SQLite.
            deadline_row = conn.execute(
                "SELECT deadline_time FROM events WHERE id=?", (int(args.gw),)
            ).fetchone()
            official_deadline = (
                str(deadline_row["deadline_time"])
                if deadline_row is not None and deadline_row["deadline_time"]
                else None
            )
            guard_cutoff = str(args.cutoff) if getattr(args, "cutoff", None) else utc_now()
            allow_late = bool(official_deadline) and (
                (parse_utc(guard_cutoff) or guard_cutoff)
                > (parse_utc(official_deadline) or guard_cutoff)
            )
            with execution.production_run_guard(
                conn,
                planning_event=int(args.gw),
                planning_cutoff=guard_cutoff,
                label="freeze_predictions",
                families=sorted(selected),
                official_deadline=official_deadline,
                allow_late=allow_late,
            ) as guard:
                run = guard.run()
                print(
                    f"execution guard: run_uuid={run.run_uuid} hard_stop={run.hard_stop_at} "
                    f"deadline_status={'LATE_FREEZE' if allow_late else 'PRE_DEADLINE'}"
                )
                code = _freeze(conn, config, args, selected)
                if code != 0:
                    guard.finish(execution.RUN_FAILED, f"_freeze returned {code}")
                return code
        finally:
            conn.close()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


def _scoring_verification(config: dict) -> tuple[object, dict]:
    """Verify the scoring rules against the newest stored official payload."""

    rules = scoring_rules.DEFAULT_SCORING_RULES
    scoring_dict, path = scoring_rules.load_stored_scoring(config_path(config, "raw_dir"))
    if not scoring_dict:
        return rules, {"verified": True, "source": "live_api_constants", "payload_path": None, "mismatches": []}
    mismatches = scoring_rules.verify_against_scoring_dict(rules, scoring_dict)
    return rules, {
        "verified": not mismatches,
        "source": "stored_official_payload",
        "payload_path": path,
        "mismatches": mismatches,
    }


DIAG_CERTIFICATION_SOURCE_SNAPSHOT_REQUIRED = "CERTIFICATION_SOURCE_SNAPSHOT_REQUIRED"


class MissingSourceSnapshot(RuntimeError):
    """Production freeze mode was entered without an immutable source connection."""


def _freeze(conn, config: dict, args, selected: set[str], source_conn=None, production: bool = False) -> int:
    """Build and persist projections.

    ``conn`` is the WRITE database (new projection runs are inserted there).
    ``source_conn`` is the database all *source* reads come from; for a
    certification that is the immutable execution snapshot.

    ``production=True`` REQUIRES an explicit ``source_conn`` and fails closed with
    ``CERTIFICATION_SOURCE_SNAPSHOT_REQUIRED`` without one, so a production
    certification can never silently fall back to reading the live fpl.db.  Legacy
    and test callers keep the permissive default ``source_conn = conn``, which is
    now an explicit non-production choice rather than an implicit default.
    """

    if production and source_conn is None:
        raise MissingSourceSnapshot(
            f"{DIAG_CERTIFICATION_SOURCE_SNAPSHOT_REQUIRED}: production freeze mode requires an "
            "explicit immutable source connection; refusing to read the live database"
        )
    if source_conn is None:
        source_conn = conn
    event = int(args.gw)
    deadline_row = conn.execute("SELECT deadline_time FROM events WHERE id=?", (event,)).fetchone()
    if deadline_row is None or deadline_row["deadline_time"] is None:
        print(f"freeze failed: no official deadline for GW{event}", file=sys.stderr)
        return 2
    deadline = str(deadline_row["deadline_time"])
    now = str(args.cutoff) if getattr(args, "cutoff", None) else utc_now()
    # Causal invariant: a planning cutoff may not lead the execution clock.
    # A cutoff 10 minutes in the future fails closed with PLANNING_CUTOFF_IN_FUTURE.
    causality.assert_causal_cutoff(now, utc_now(), label=f"GW{event} freeze")
    deadline_status = "LATE_FREEZE" if (parse_utc(now) or now) > (parse_utc(deadline) or now) else "PRE_DEADLINE"
    cutoff = now
    source_hash = analytics.source_snapshot_sha256()

    entry_id = config.get("fpl_entry_id")
    context = get_planning_context(
        source_conn, int(entry_id), event, as_of=cutoff, season=config.get("season"),
        official_price_stale_after_hours=config["report"].get("official_price_stale_after_hours"),
    )
    if context.health["status"] == "FAIL":
        print("freeze failed: PlanningContext health FAIL", file=sys.stderr)
        for reason in context.health["fail_reasons"]:
            print(f"  - {reason}", file=sys.stderr)
        return 2

    context_ref = analytics.planning_context_reference(context)
    rules, rules_verification = _scoring_verification(config)
    if not rules_verification["verified"]:
        print("freeze failed: scoring-rules drift against the official payload", file=sys.stderr)
        for mismatch in rules_verification["mismatches"]:
            print(f"  - {mismatch}", file=sys.stderr)
        return 2

    prepared: dict[str, object] = {}
    readiness: dict[str, dict] = {}

    if "baseline" in selected or "minutes" in selected:
        minutes_config = minutes_model.MinutesModelConfig()
        minutes_rows = minutes_model.build_minutes_predictions(source_conn, event, cutoff, minutes_config)
        prepared["minutes"] = (minutes_config, minutes_rows)
        readiness["minutes"] = minutes_model.readiness_summary(
            context, minutes_rows, deadline_status=deadline_status, data_cutoff=cutoff, deadline=deadline
        )
    if "minutes_coherent" in selected:
        minutes_config = minutes_model.MinutesModelConfig()
        coherence_config = minutes_coherence.MinutesCoherenceConfig()
        try:
            coherent_rows, coherence_records = minutes_coherence.build_minutes_predictions_coherent(
                source_conn, event, cutoff, minutes_config, coherence_config
            )
        except minutes_coherence.TeamCoherenceError as exc:
            print(f"freeze failed: team-coherence impossible: {exc}", file=sys.stderr)
            return 3
        prepared["minutes_coherent"] = (minutes_config, coherence_config, coherent_rows, coherence_records)
        base_readiness = minutes_model.readiness_summary(
            context, coherent_rows, deadline_status=deadline_status, data_cutoff=cutoff, deadline=deadline
        )
        coherence_gate = minutes_coherence.coherence_readiness(coherence_records, coherence_config)
        readiness["minutes_coherent"] = {
            "status": "FAIL" if (base_readiness["status"] == "FAIL" or coherence_gate["status"] == "FAIL") else "WARN",
            "fail_reasons": base_readiness["fail_reasons"] + coherence_gate["fail_reasons"],
            "warn_reasons": base_readiness["warn_reasons"],
            "counts": {**base_readiness["counts"], **coherence_gate["counts"]},
            "coherence": coherence_gate,
        }
    if "minutes_positional" in selected:
        minutes_config = minutes_model.MinutesModelConfig()
        coherence_config = minutes_coherence.MinutesCoherenceConfig()
        try:
            positional_rows, positional_records = minutes_coherence.build_minutes_predictions_coherent(
                source_conn, event, cutoff, minutes_config, coherence_config, positional=True
            )
        except minutes_coherence.TeamCoherenceError as exc:
            print(f"freeze failed: positional team-coherence impossible: {exc}", file=sys.stderr)
            return 3
        prepared["minutes_positional"] = (minutes_config, coherence_config, positional_rows, positional_records)
        base_readiness = minutes_model.readiness_summary(
            context, positional_rows, deadline_status=deadline_status, data_cutoff=cutoff, deadline=deadline
        )
        coherence_gate = minutes_coherence.coherence_readiness(positional_records, coherence_config)
        readiness["minutes_positional"] = {
            "status": "FAIL" if (base_readiness["status"] == "FAIL" or coherence_gate["status"] == "FAIL") else "WARN",
            "fail_reasons": base_readiness["fail_reasons"] + coherence_gate["fail_reasons"],
            "warn_reasons": base_readiness["warn_reasons"],
            "counts": {**base_readiness["counts"], **coherence_gate["counts"]},
            "coherence": coherence_gate,
        }
    if "minutes_substitution" in selected:
        sub_minutes_config = minutes_model.MinutesModelConfig()
        sub_coherence_config = minutes_coherence.MinutesCoherenceConfig()
        sub_config = substitution_model.SubstitutionConfig()
        try:
            substitution_rows, substitution_profiles = substitution_model.build_minutes_predictions_substitution_coherent(
                source_conn, event, cutoff, sub_minutes_config, sub_coherence_config, sub_config
            )
        except ValueError as exc:
            print(f"freeze failed: substitution-coherent minutes impossible: {exc}", file=sys.stderr)
            return 3
        prepared["minutes_substitution"] = (sub_minutes_config, sub_config, substitution_rows, substitution_profiles)
        base_readiness = minutes_model.readiness_summary(
            context, substitution_rows, deadline_status=deadline_status, data_cutoff=cutoff, deadline=deadline
        )
        incoherent = [p for p in substitution_profiles if p.get("status") != "COHERENT"]
        readiness["minutes_substitution"] = {
            "status": "FAIL" if (base_readiness["status"] == "FAIL" or incoherent) else "WARN",
            "fail_reasons": base_readiness["fail_reasons"] + [
                f"SUBSTITUTION_PROFILE_INCOHERENT: fixture {p['fixture_id']} team {p['team_id']}" for p in incoherent
            ],
            "warn_reasons": base_readiness["warn_reasons"],
            "counts": {**base_readiness["counts"], "profiles": len(substitution_profiles)},
            "identities": {
                "max_abs_p_start_minus_11": max(
                    (abs(p["identities"]["p_start_total"] - 11.0) for p in substitution_profiles), default=0.0
                ),
                "max_abs_expected_minutes_minus_990": max(
                    (abs(p["identities"]["expected_minutes"] - 990.0) for p in substitution_profiles), default=0.0
                ),
                "max_abs_exit_minus_entry": max(
                    (abs(p["identities"]["exit_mass"] - p["identities"]["entry_mass"]) for p in substitution_profiles),
                    default=0.0,
                ),
                "max_abs_mass_residual": max(
                    (p.get("max_abs_constraint_residual", 0.0) for p in substitution_profiles), default=0.0
                ),
            },
        }
    if "minutes_joint" in selected:
        joint_config = joint_minutes.JointMinutesConfig()
        sub_config = substitution_model.SubstitutionConfig()
        try:
            joint_rows, joint_profiles = joint_minutes.build_minutes_v15(
                source_conn, event, cutoff, joint_config=joint_config, sub_config=sub_config
            )
        except ValueError as exc:
            print(f"freeze failed: joint minutes impossible: {exc}", file=sys.stderr)
            return 3
        prepared["minutes_joint"] = (joint_config, joint_rows, joint_profiles)
        errs = [abs(r["p_start_absolute_error"]) for r in joint_rows]
        worst = max(errs) if errs else 0.0
        readiness["minutes_joint"] = {
            "status": "FAIL" if worst > 0.05 else "WARN",
            "fail_reasons": ([f"START_MARGINAL_INTEGRATION_BIAS: max |target - integrated| {worst:.4f}"]
                             if worst > 0.05 else []),
            "warn_reasons": [f"SUBSTITUTION_PRIOR_EARLY_SEASON: {joint_profiles[0]['evidence_matches']} team-fixtures"],
            "counts": {"profiles": len(joint_profiles), "draws": joint_config.integration_draws},
            "identities": {
                "max_abs_start_error": worst,
                "mean_abs_start_error": (sum(errs) / len(errs)) if errs else 0.0,
            },
            # Section 18: realised cameo/zero/long-start marginals by position,
            # reported against the audited substitution evidence shape.  These are
            # diagnostics, never per-player tuning targets.
            "output_diagnostics": joint_minutes.minutes_output_diagnostics(joint_rows),
        }
    if "team" in selected:
        team_config = team_model.TeamStrengthConfig()
        team_rows, _meta = team_model.build_team_fixture_projections(source_conn, event, cutoff, team_config)
        naive_team_rows = team_model.build_naive_team_projections(source_conn, event, cutoff, team_config)
        prepared["team"] = (team_config, team_rows, naive_team_rows)
        readiness["team"] = team_model.readiness_summary(
            context, team_rows, deadline_status=deadline_status, data_cutoff=cutoff, deadline=deadline
        )
    if "rates" in selected:
        rate_config = player_rates.PlayerRatesConfig()
        rate_rows = player_rates.build_player_rate_projections(source_conn, event, cutoff, rate_config)
        rate_baselines = player_rates.build_naive_rate_baselines(source_conn, event, cutoff, rate_config)
        prepared["rates"] = (rate_config, rate_rows, rate_baselines)
        readiness["rates"] = player_rates.readiness_summary(
            context, rate_rows, deadline_status=deadline_status, data_cutoff=cutoff, deadline=deadline
        )

    explicit_xpts = any([args.xpts_minutes_run, args.xpts_team_run, args.xpts_rate_run])
    standalone_mc = "monte_carlo" in selected and "xpts" not in selected
    if standalone_mc:
        missing = [name for name, value in (
            ("--xpts-run", args.xpts_run), ("--xpts-minutes-run", args.xpts_minutes_run),
            ("--xpts-team-run", args.xpts_team_run), ("--xpts-rate-run", args.xpts_rate_run),
        ) if not value]
        if missing:
            print(f"configuration error: standalone Monte Carlo requires {', '.join(missing)}", file=sys.stderr)
            return 2
    if "xpts" in selected and explicit_xpts:
        missing = [name for name, value in (
            ("--xpts-minutes-run", args.xpts_minutes_run),
            ("--xpts-team-run", args.xpts_team_run),
            ("--xpts-rate-run", args.xpts_rate_run),
        ) if not value]
        if missing:
            print(f"configuration error: xPts requires {', '.join(missing)}", file=sys.stderr)
            return 2

    if args.dry_run:
        parts = []
        for name in sorted(readiness):
            row_count = 0
            if name == "minutes":
                row_count = len(prepared["minutes"][1])  # type: ignore[index]
            elif name == "minutes_coherent":
                row_count = len(prepared["minutes_coherent"][2])  # type: ignore[index]
            elif name == "minutes_positional":
                row_count = len(prepared["minutes_positional"][2])  # type: ignore[index]
            elif name == "minutes_substitution":
                row_count = len(prepared["minutes_substitution"][2])  # type: ignore[index]
            elif name == "minutes_joint":
                row_count = len(prepared["minutes_joint"][1])  # type: ignore[index]
            elif name == "team":
                row_count = len(prepared["team"][1])  # type: ignore[index]
            elif name == "rates":
                row_count = len(prepared["rates"][1])  # type: ignore[index]
            parts.append(f"{name}={readiness[name]['status']}({row_count} rows)")
        if "xpts" in selected:
            if explicit_xpts:
                built = _build_xpts(conn, event, cutoff, {
                    "minutes": args.xpts_minutes_run, "team": args.xpts_team_run,
                    "team_baseline": args.xpts_team_baseline_run, "rate": args.xpts_rate_run,
                }, rules, deadline_status=deadline_status, deadline=deadline)
                parts.append(f"xpts={built['readiness']['status']}({len(built['rows'])} rows)")
            else:
                parts.append("xpts=DEFERRED(needs a real synchronized run)")
        print(f"dry-run: GW{event} {deadline_status} {', '.join(parts)} ctx_ref={context_ref[:20]}…")
        return 0

    failed = {name: result for name, result in readiness.items() if result["status"] == "FAIL"}
    if failed:
        print("freeze failed: analytics readiness FAIL", file=sys.stderr)
        for name, result in sorted(failed.items()):
            for reason in result["fail_reasons"]:
                print(f"  - {name}: {reason}", file=sys.stderr)
        return 3

    run_ids: dict[str, int] = {}
    counts: dict[str, object] = {}
    xpts_meta: dict | None = None
    xpts_readiness: dict | None = None
    used_xpts_inputs: dict | None = None
    mc_meta: dict | None = None
    try:
        with conn:
            run_ids, counts = _freeze_components(
                conn, context, context_ref, prepared, selected, cutoff, deadline_status, event, source_hash
            )
            if "xpts" in selected:
                if explicit_xpts:
                    xpts_inputs = {
                        "minutes": args.xpts_minutes_run, "team": args.xpts_team_run,
                        "team_baseline": args.xpts_team_baseline_run, "rate": args.xpts_rate_run,
                    }
                else:
                    xpts_inputs = {
                        "minutes": (run_ids.get("minutes_joint")
                                    or run_ids.get("minutes_substitution")
                                    or run_ids.get("minutes_positional")
                                    or run_ids.get("minutes_coherent")
                                    or run_ids.get("minutes")),
                        "team": run_ids.get("team"),
                        "team_baseline": run_ids.get("team_baseline"), "rate": run_ids.get("rates"),
                    }
                    if not all(xpts_inputs.values()):
                        raise _ReadinessFailure(
                            "xpts", ["xPts needs minutes/team/rate runs: freeze them together or pass --xpts-*-run"]
                        )
                used_xpts_inputs = dict(xpts_inputs)
                strict_coherence = bool(run_ids.get("minutes_joint")) or bool(run_ids.get("minutes_substitution")) or bool(run_ids.get("minutes_coherent")) or bool(run_ids.get("minutes_positional")) or bool(
                    getattr(args, "xpts_minutes_run", None)
                    and _run_is_coherent(conn, args.xpts_minutes_run)
                )
                built = _build_xpts(conn, event, cutoff, xpts_inputs, rules,
                                    deadline_status=deadline_status, deadline=deadline,
                                    strict_coherence=strict_coherence)
                xpts_readiness = built["readiness"]
                if xpts_readiness["status"] == "FAIL":
                    raise _ReadinessFailure("xpts", xpts_readiness["fail_reasons"])
                xpts_run_id = analytics.create_projection_run(
                    conn, model_family=analytics.XPTS_MODEL_FAMILY, model_version=xpts.XPTS_MODEL_VERSION,
                    planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
                    scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
                    config_hash=built["config_hash"], deadline_status=deadline_status, source_snapshot_sha256=source_hash,
                )
                for row in built["rows"]:
                    analytics.freeze_xpts_projection(
                        conn, xpts_run_id,
                        player_id=int(row["player_id"]), fixture_id=int(row["fixture_id"]), event=event,
                        team_id=int(row["team_id"]), opponent_id=int(row["opponent_id"]),
                        position=str(row["position"]),
                        minutes_run_id=int(xpts_inputs["minutes"]), team_run_id=int(xpts_inputs["team"]),
                        rate_run_id=int(xpts_inputs["rate"]),
                        payload={key: value for key, value in row.items()
                                 if key not in {"player_id", "fixture_id", "event", "team_id", "opponent_id", "position"}},
                        model_version=xpts.XPTS_MODEL_VERSION,
                        scoring_rules_version=scoring_rules.SCORING_RULES_VERSION,
                    )
                analytics.finish_projection_run(conn, xpts_run_id, "complete")
                run_ids["xpts"] = xpts_run_id
                counts["xpts_rows"] = len(built["rows"])
                xpts_meta = {
                    "meta": built["meta"],
                    "coherence": built["coherence"],
                    "config_hash": built["config_hash"],
                    "input_run_ids": {k: int(v) for k, v in xpts_inputs.items() if v},
                }
            if "monte_carlo" in selected:
                if standalone_mc:
                    run_ids["xpts"] = int(args.xpts_run)
                    used_xpts_inputs = {
                        "minutes": args.xpts_minutes_run, "team": args.xpts_team_run,
                        "rate": args.xpts_rate_run,
                    }
                if not used_xpts_inputs or not run_ids.get("xpts"):
                    raise _ReadinessFailure(
                        "monte_carlo", ["Monte Carlo needs an xPts run in the same freeze (or pass xpts inputs)"]
                    )
                coherence_problems = monte_carlo.validate_input_run_coherence(
                    conn, xpts_run_id=run_ids["xpts"],
                    minutes_run_id=int(used_xpts_inputs["minutes"]),
                    team_run_id=int(used_xpts_inputs["team"]),
                    rate_run_id=int(used_xpts_inputs["rate"]) if used_xpts_inputs.get("rate") else None,
                )
                if coherence_problems:
                    raise _ReadinessFailure("monte_carlo", coherence_problems)
                # Certification runs must actually MEASURE occupancy; the
                # structural argument is not certification evidence.
                mc_config = monte_carlo.MonteCarloConfig(
                    simulations=int(args.mc_simulations), occupancy_audit=True,
                    calibration_states=int(args.mc_calibration_states),
                    calibration_max_iterations=int(args.mc_calibration_iterations),
                )
                fixtures = monte_carlo.load_fixture_inputs(
                    conn, event=event, xpts_run_id=run_ids["xpts"],
                    minutes_run_id=int(used_xpts_inputs["minutes"]), team_run_id=int(used_xpts_inputs["team"]),
                )
                mc_result = monte_carlo.simulate(fixtures, mc_config, rules)
                mc_readiness = monte_carlo.readiness_summary(
                    fixtures, mc_result, mc_config, deadline_status=deadline_status,
                    data_cutoff=cutoff, deadline=deadline,
                )
                if mc_readiness["status"] == "FAIL":
                    raise _ReadinessFailure("monte_carlo", mc_readiness["fail_reasons"])
                mc_run_id = analytics.create_projection_run(
                    conn, model_family=monte_carlo.MONTE_CARLO_MODEL_FAMILY,
                    model_version=monte_carlo.MONTE_CARLO_MODEL_VERSION, planning_event=event,
                    planning_context_hash=context_ref, data_cutoff=cutoff, scouting_cutoff=context.scouting_cutoff,
                    official_run_ids=context.official_runs, config_hash=mc_config.config_hash(),
                    random_seed=mc_config.seed, deadline_status=deadline_status, source_snapshot_sha256=source_hash,
                )
                player_meta = {int(r["player_id"]): r for r in analytics.xpts_projections(conn, run_ids["xpts"])}
                for summary in mc_result["summaries"]:
                    source = player_meta.get(int(summary["player_id"]))
                    if source is None:
                        continue
                    analytics.freeze_monte_carlo_distribution(
                        conn, mc_run_id, player_id=int(summary["player_id"]), fixture_id=int(summary["fixture_id"]),
                        event=event, team_id=int(source["team_id"]), opponent_id=int(source["opponent_id"]),
                        position=str(source["position"]), xpts_run_id=run_ids["xpts"],
                        minutes_run_id=int(used_xpts_inputs["minutes"]), team_run_id=int(used_xpts_inputs["team"]),
                        rate_run_id=int(used_xpts_inputs["rate"]),
                        payload=summary, model_version=monte_carlo.MONTE_CARLO_MODEL_VERSION,
                    )
                # Calibration provenance is assembled and VALIDATED before the run
                # is marked complete, so a certifiable Monte Carlo run can never be
                # recorded without a recoverable calibration-state identity.
                calibration_provenance = monte_carlo.calibration_provenance(mc_config)
                monte_carlo.require_calibration_provenance(
                    calibration_provenance, model_version=monte_carlo.MONTE_CARLO_MODEL_VERSION
                )
                analytics.finish_projection_run(conn, mc_run_id, "complete")
                run_ids["monte_carlo"] = mc_run_id
                counts["monte_carlo_rows"] = len(mc_result["summaries"])
                mc_meta = {
                    "model_version": monte_carlo.MONTE_CARLO_MODEL_VERSION,
                    "config_hash": mc_config.config_hash(),
                    **calibration_provenance,
                    "seed": mc_config.seed,
                    "simulations": mc_config.simulations,
                    "max_substitute_entrants": mc_config.max_substitute_entrants,
                    "substitution_limit_source": monte_carlo.SUBSTITUTION_LIMIT_SOURCE,
                    "occupancy_audit": bool(mc_config.occupancy_audit),
                    "occupancy_violations": int(mc_result.get("occupancy_violations") or 0),
                    "mc_primitives_verbatim": bool(mc_readiness.get("mc_primitives_verbatim")),
                    "source_snapshot_sha256": source_hash,
                    "readiness": mc_readiness,
                    "team_minutes": {
                        "mean": sum(mc_result["team_minutes"]) / max(1, len(mc_result["team_minutes"])),
                        "min": min(mc_result["team_minutes"], default=0.0),
                        "max": max(mc_result["team_minutes"], default=0.0),
                        "n_worlds": len(mc_result["team_minutes"]),
                    },
                }
    except _ReadinessFailure as failure:
        print(f"freeze failed: {failure.name} readiness FAIL (nothing written)", file=sys.stderr)
        for reason in failure.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 3

    if deadline_status == "LATE_FREEZE":
        print(f"LATE_FREEZE recorded: deadline {deadline} passed before execution; the rows are a late freeze.")

    artifact_dir = Path(args.out_dir) if args.out_dir else config_path(config, "exports_dir") / "predictions"
    artifact = artifact_dir / f"gw{event:02d}" / f"prediction_freeze_run{max(run_ids.values())}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    provenance = analytics.git_provenance()
    provenance["source_snapshot_sha256"] = source_hash
    artifact.write_text(
        json.dumps({
            "schema": "fpl_brain.prediction_freeze.v3",
            "event": event, "deadline": deadline, "deadline_status": deadline_status,
            "data_cutoff": cutoff, "scouting_cutoff": context.scouting_cutoff,
            "planning_context_hash": context_ref, "families": sorted(selected),
            "run_ids": run_ids, "counts": counts,
            "provenance": provenance,
            "source_snapshot_sha256": source_hash,
            "scoring_rules": {
                "version": scoring_rules.SCORING_RULES_VERSION,
                "verification": rules_verification,
                "scoring_hash": rules.scoring_hash(),
            },
            "model_versions": {
                "minutes_v1": minutes_model.MINUTES_MODEL_VERSION,
                "minutes_coherent_v1_2": minutes_coherence.MINUTES_COHERENT_MODEL_VERSION,
                "minutes_positional_v1_3": minutes_coherence.MINUTES_POSITIONAL_MODEL_VERSION,
                "minutes_substitution_v1_4": substitution_model.MINUTES_SUBSTITUTION_MODEL_VERSION,
                "minutes_joint_v1_5": joint_minutes.JOINT_MINUTES_MODEL_VERSION,
                "monte_carlo_v1": monte_carlo.MONTE_CARLO_MODEL_VERSION,
                "team_strength_v1": team_model.TEAM_MODEL_VERSION,
                "team_baseline": team_model.TEAM_BASELINE_MODEL_VERSION,
                "player_rates_v1": player_rates.PLAYER_RATE_MODEL_VERSION,
                "player_rate_baseline": player_rates.PLAYER_RATE_BASELINE_MODEL_VERSION,
                "xpts_v1": xpts.XPTS_MODEL_VERSION,
            },
            "readiness": {**readiness, **({"xpts": xpts_readiness} if xpts_readiness else {})},
            "xpts": xpts_meta,
            "monte_carlo": mc_meta,
            "no_manager_decisions": True,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"freeze complete: GW{event} {deadline_status} families={sorted(selected)} runs={run_ids} "
        f"readiness={ {k: v['status'] for k, v in {**readiness, **({'xpts': xpts_readiness} if xpts_readiness else {})}.items()} } "
        f"artifact={artifact}"
    )
    return 0


def _freeze_components(conn, context, context_ref, prepared, selected, cutoff, deadline_status, event, source_hash):
    """Create and freeze the component runs; returns (run_ids, counts)."""

    run_ids: dict[str, int] = {}
    counts: dict[str, object] = {}
    if "baseline" in selected:
        baseline_run_id, baseline_counts = analytics.freeze_baselines_for_event(
            conn, context, cutoff, event=event, deadline_status=deadline_status
        )
        run_ids["baseline"] = baseline_run_id
        counts["baseline_rows"] = baseline_counts
    if "minutes" in selected:
        minutes_config, minutes_rows = prepared["minutes"]  # type: ignore[misc]
        minutes_run_id = analytics.create_projection_run(
            conn, model_family=analytics.MINUTES_MODEL_FAMILY, model_version=minutes_model.MINUTES_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=minutes_config.config_hash(), deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in minutes_rows:
            analytics.freeze_prediction(
                conn, minutes_run_id, kind=analytics.MINUTES_V1_KIND, player_id=int(row["player_id"]),
                event=event, fixture_id=int(row["fixture_id"]),
                payload={k: v for k, v in row.items() if k not in {"player_id", "fixture_id", "event"}},
                model_version=minutes_model.MINUTES_MODEL_VERSION,
            )
        analytics.finish_projection_run(conn, minutes_run_id, "complete")
        run_ids["minutes"] = minutes_run_id
        counts["minutes_rows"] = len(minutes_rows)
    if "minutes_coherent" in selected:
        minutes_config, coherence_config, coherent_rows, coherence_records = prepared["minutes_coherent"]  # type: ignore[misc]
        coherence_run_id = analytics.create_projection_run(
            conn, model_family=analytics.MINUTES_MODEL_FAMILY, model_version=minutes_coherence.MINUTES_COHERENT_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=analytics.canonical_hash(
                {"minutes": minutes_config.config_hash(), "coherence": coherence_config.config_hash()}
            ),
            deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in coherent_rows:
            analytics.freeze_prediction(
                conn, coherence_run_id, kind=analytics.MINUTES_V1_KIND, player_id=int(row["player_id"]),
                event=event, fixture_id=int(row["fixture_id"]),
                payload={k: v for k, v in row.items() if k not in {"player_id", "fixture_id", "event"}},
                model_version=minutes_coherence.MINUTES_COHERENT_MODEL_VERSION,
            )
        for record in coherence_records:
            analytics.freeze_team_minutes_coherence(conn, coherence_run_id, record)
        analytics.finish_projection_run(conn, coherence_run_id, "complete")
        run_ids["minutes_coherent"] = coherence_run_id
        counts["minutes_coherent_rows"] = len(coherent_rows)
        counts["coherence_records"] = len(coherence_records)
    if "minutes_positional" in selected:
        minutes_config, coherence_config, positional_rows, positional_records = prepared["minutes_positional"]  # type: ignore[misc]
        positional_run_id = analytics.create_projection_run(
            conn, model_family=analytics.MINUTES_MODEL_FAMILY,
            model_version=minutes_coherence.MINUTES_POSITIONAL_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=analytics.canonical_hash(
                {"minutes": minutes_config.config_hash(), "coherence": coherence_config.config_hash(),
                 "positional": True}
            ),
            deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in positional_rows:
            analytics.freeze_prediction(
                conn, positional_run_id, kind=analytics.MINUTES_V1_KIND, player_id=int(row["player_id"]),
                event=event, fixture_id=int(row["fixture_id"]),
                payload={k: v for k, v in row.items() if k not in {"player_id", "fixture_id", "event"}},
                model_version=minutes_coherence.MINUTES_POSITIONAL_MODEL_VERSION,
            )
        for record in positional_records:
            analytics.freeze_team_minutes_coherence(conn, positional_run_id, record)
        analytics.finish_projection_run(conn, positional_run_id, "complete")
        run_ids["minutes_positional"] = positional_run_id
        counts["minutes_positional_rows"] = len(positional_rows)
    if "minutes_substitution" in selected:
        sub_minutes_config, sub_config, substitution_rows, substitution_profiles = prepared["minutes_substitution"]  # type: ignore[misc]
        substitution_run_id = analytics.create_projection_run(
            conn, model_family=analytics.MINUTES_MODEL_FAMILY,
            model_version=substitution_model.MINUTES_SUBSTITUTION_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=analytics.canonical_hash(
                {"minutes": sub_minutes_config.config_hash(), "substitution": sub_config.config_hash()}
            ),
            deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in substitution_rows:
            analytics.freeze_prediction(
                conn, substitution_run_id, kind=analytics.MINUTES_V1_KIND, player_id=int(row["player_id"]),
                event=event, fixture_id=int(row["fixture_id"]),
                payload={k: v for k, v in row.items() if k not in {"player_id", "fixture_id", "event"}},
                model_version=substitution_model.MINUTES_SUBSTITUTION_MODEL_VERSION,
            )
        for profile in substitution_profiles:
            analytics.freeze_team_substitution_profile(conn, substitution_run_id, profile)
        analytics.finish_projection_run(conn, substitution_run_id, "complete")
        run_ids["minutes_substitution"] = substitution_run_id
        counts["minutes_substitution_rows"] = len(substitution_rows)
        counts["substitution_profiles"] = len(substitution_profiles)
    if "minutes_joint" in selected:
        joint_config, joint_rows, joint_profiles = prepared["minutes_joint"]  # type: ignore[misc]
        joint_run_id = analytics.create_projection_run(
            conn, model_family=analytics.MINUTES_MODEL_FAMILY,
            model_version=joint_minutes.JOINT_MINUTES_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=joint_config.config_hash(), deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in joint_rows:
            analytics.freeze_prediction(
                conn, joint_run_id, kind=analytics.MINUTES_V1_KIND, player_id=int(row["player_id"]),
                event=event, fixture_id=int(row["fixture_id"]),
                payload={k: v for k, v in row.items() if k not in {"player_id", "fixture_id", "event"}},
                model_version=joint_minutes.JOINT_MINUTES_MODEL_VERSION,
            )
        for profile in joint_profiles:
            analytics.freeze_team_substitution_profile(conn, joint_run_id, profile)
        analytics.finish_projection_run(conn, joint_run_id, "complete")
        run_ids["minutes_joint"] = joint_run_id
        counts["minutes_joint_rows"] = len(joint_rows)
        counts["joint_substitution_profiles"] = len(joint_profiles)
    if "team" in selected:
        team_config, team_rows, naive_team_rows = prepared["team"]  # type: ignore[misc]
        team_run_id = analytics.create_projection_run(
            conn, model_family=analytics.TEAM_MODEL_FAMILY, model_version=team_model.TEAM_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=team_config.config_hash(), deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in team_rows:
            analytics.freeze_team_fixture_projection(
                conn, team_run_id, fixture_id=int(row["fixture_id"]), event=event, team_id=int(row["team_id"]),
                opponent_id=int(row["opponent_id"]), venue=str(row["venue"]),
                payload={k: v for k, v in row.items() if k not in {"fixture_id", "team_id", "opponent_id", "venue", "event"}},
                model_version=team_model.TEAM_MODEL_VERSION,
            )
        analytics.finish_projection_run(conn, team_run_id, "complete")
        run_ids["team"] = team_run_id
        counts["team_rows"] = len(team_rows)

        team_baseline_run_id = analytics.create_projection_run(
            conn, model_family=analytics.TEAM_BASELINE_MODEL_FAMILY, model_version=team_model.TEAM_BASELINE_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=team_config.config_hash(), deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in naive_team_rows:
            analytics.freeze_team_fixture_projection(
                conn, team_baseline_run_id, fixture_id=int(row["fixture_id"]), event=event,
                team_id=int(row["team_id"]), opponent_id=int(row["opponent_id"]), venue=str(row["venue"]),
                payload={k: v for k, v in row.items() if k not in {"fixture_id", "team_id", "opponent_id", "venue", "event"}},
                model_version=team_model.TEAM_BASELINE_MODEL_VERSION,
            )
        analytics.finish_projection_run(conn, team_baseline_run_id, "complete")
        run_ids["team_baseline"] = team_baseline_run_id
        counts["team_baseline_rows"] = len(naive_team_rows)
    if "rates" in selected:
        rate_config, rate_rows, rate_baselines = prepared["rates"]  # type: ignore[misc]
        rates_run_id = analytics.create_projection_run(
            conn, model_family=analytics.PLAYER_RATES_MODEL_FAMILY, model_version=player_rates.PLAYER_RATE_MODEL_VERSION,
            planning_event=event, planning_context_hash=context_ref, data_cutoff=cutoff,
            scouting_cutoff=context.scouting_cutoff, official_run_ids=context.official_runs,
            config_hash=rate_config.config_hash(), deadline_status=deadline_status, source_snapshot_sha256=source_hash,
        )
        for row in rate_rows:
            analytics.freeze_player_rate_projection(
                conn, rates_run_id, player_id=int(row["player_id"]), component=str(row["component"]), event=event,
                payload={k: v for k, v in row.items() if k not in {"player_id", "component", "event"}},
                model_version=player_rates.PLAYER_RATE_MODEL_VERSION,
            )
        analytics.finish_projection_run(conn, rates_run_id, "complete")
        run_ids["rates"] = rates_run_id
        counts["rate_rows"] = len(rate_rows)

        for key, kind in (("raw_baseline", "RAW_CURRENT_RATE"), ("prior_baseline", "PRIOR_ONLY_RATE")):
            baseline_run_id = analytics.create_projection_run(
                conn, model_family=analytics.PLAYER_RATE_BASELINE_MODEL_FAMILY,
                model_version=player_rates.PLAYER_RATE_BASELINE_MODEL_VERSION, planning_event=event,
                planning_context_hash=context_ref, data_cutoff=cutoff, scouting_cutoff=context.scouting_cutoff,
                official_run_ids=context.official_runs, config_hash=rate_config.config_hash(),
                deadline_status=deadline_status, source_snapshot_sha256=source_hash,
            )
            baseline_rows = rate_baselines[kind]
            for row in baseline_rows:
                analytics.freeze_player_rate_projection(
                    conn, baseline_run_id, player_id=int(row["player_id"]), component=str(row["component"]),
                    event=event,
                    payload={k: v for k, v in row.items() if k not in {"player_id", "component", "event", "baseline_kind"}},
                    model_version=player_rates.PLAYER_RATE_BASELINE_MODEL_VERSION,
                )
            analytics.finish_projection_run(conn, baseline_run_id, "complete")
            run_ids[key] = baseline_run_id
            counts[f"{key}_rows"] = len(baseline_rows)
    return run_ids, counts


def _run_is_coherent(conn, run_id) -> bool:
    """True when a minutes run was produced by the team-coherence layer."""

    if not run_id:
        return False
    run = analytics.get_projection_run(conn, int(run_id))
    if not run:
        return False
    if str(run.get("model_version") or "").startswith(("minutes_v1.2", "minutes_v1.3", "minutes_v1.4", "minutes_v1.5")):
        return True
    return bool(analytics.team_minutes_coherence_records(conn, int(run_id)))


def _build_xpts(conn, event: int, cutoff: str, inputs: dict, rules, *,
                deadline_status: str = "PRE_DEADLINE", deadline: str | None = None,
                strict_coherence: bool = False) -> dict:
    """Build xPts rows from explicit input runs and compute readiness."""

    config = xpts.XPtsConfig()
    built = xpts.build_xpts_projections(
        conn, event=event, cutoff=cutoff,
        minutes_run_id=int(inputs["minutes"]), team_run_id=int(inputs["team"]),
        team_baseline_run_id=int(inputs["team_baseline"]) if inputs.get("team_baseline") else None,
        rate_run_id=int(inputs["rate"]), config=config, rules=rules,
    )
    strict = strict_coherence or bool(built["meta"].get("input_team_coherence"))
    readiness = xpts.readiness_summary(
        None, built["rows"], built["coherence"], deadline_status=deadline_status,
        data_cutoff=cutoff, deadline=deadline, rules_verified=True,
        strict_coherence=strict, config=config,
    )
    return {
        "rows": built["rows"], "meta": built["meta"], "coherence": built["coherence"],
        "readiness": readiness, "config_hash": config.config_hash(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
