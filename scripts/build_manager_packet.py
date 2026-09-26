#!/usr/bin/env python3
"""Phase 6A — build the fixed-15 manager policy evaluation and decision packet.

Descriptive validation only: no lineup change, transfer or chip is executed.
Consumes the canonical PlanningContext and the FROZEN Phase-5 predictive runs.

PE-9: ``manager_worlds.build_manager_worlds`` is a PREDICTIVE-LOAD boundary, so this
command resolves the authoritative certified GENERATION and forwards it to that load.
The run ids the packet names are READ from the generation's digest-verified manifest; a
supplied run id that is not the recorded one is refused, and there is no GW4, run-id or
generation default: a packet built from uncertified predictions is never written.  The
published provenance exposes the ``generation_id``, the exact certified run identities,
the pinned snapshot identity, the PE-8 evidence references and the manager packet digest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import certified_bundle, manager_lineup, manager_worlds
from fpl_brain import generation_store as gs
from fpl_brain import packet as packet_mod
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE6A_VERSION = "phase6a_manager_lineup_v1.0.0"

#: The families the manager world load READS, in the manifest's own spelling.  The
#: Monte Carlo run is not read by this load (the shared worlds are regenerated), so it
#: is carried only to name the certified generation the packet describes.
MANAGER_WORLD_FAMILIES = ("minutes_v1", "team_strength_v1", "xpts_v1")


class CertificationMismatch(ValueError):
    """A caller-supplied event or run id is not the one the certified generation records."""


def certified_run_ids(
    generation,
    *,
    event: int,
    minutes_run: int | None = None,
    xpts_run: int | None = None,
    team_run: int | None = None,
    monte_carlo_run: int | None = None,
) -> dict[str, int]:
    """The CERTIFIED run ids for one event, with every supplied id validated.

    The certified generation is authoritative for the predictive world: the run ids are
    the ones its manifest records for this event, so a supplied id that is not the
    recorded one -- or an id for a family the manifest does not record -- is refused
    rather than preferred, defaulted or silently ignored.
    """

    recorded = {
        str(family): int(run_id) for family, run_id in generation.runs_for(int(event)).items()
    }
    if not recorded:
        raise CertificationMismatch(
            f"GW{int(event)}: generation {generation.generation_id} names no run for this event"
        )
    supplied = {
        "minutes_v1": minutes_run,
        "team_strength_v1": team_run,
        "xpts_v1": xpts_run,
        "monte_carlo_v1": monte_carlo_run,
    }
    for family, value in sorted(supplied.items()):
        if value is None:
            continue
        if family not in recorded:
            raise CertificationMismatch(
                f"GW{int(event)}: the certified generation records no {family} run, so the supplied "
                f"{family} run {int(value)} cannot be validated against it"
            )
        if int(recorded[family]) != int(value):
            raise CertificationMismatch(
                f"GW{int(event)}: the supplied {family} run {int(value)} is not the run the certified "
                f"generation records for this event ({recorded[family]}); the generation is "
                "authoritative"
            )
    missing = [family for family in MANAGER_WORLD_FAMILIES if family not in recorded]
    if missing:
        raise CertificationMismatch(
            f"GW{int(event)}: the certified generation names no run for {sorted(missing)}; a "
            "predictive load is authorised by the exact certified run ids and never by a partial "
            "record"
        )
    resolved = {family: recorded[family] for family in MANAGER_WORLD_FAMILIES}
    if "monte_carlo_v1" in recorded:
        resolved["monte_carlo_v1"] = recorded["monte_carlo_v1"]
    return resolved


def certified_model_versions(generation, *, event: int) -> dict[str, str]:
    """The model versions the certified generation records for one event.

    Read from the manifest rather than restated, so the packet names the versions of the
    generation it actually consumed.
    """

    return {
        str(family): str(version)
        for family, version in sorted(generation.model_versions_by_event.get(int(event), {}).items())
    }


def certified_snapshot_provenance(generation) -> dict[str, object]:
    """The pinned snapshot identity the packet reports, straight from the manifest."""

    snapshot = dict(generation.snapshot)
    return {
        "generation_id": generation.generation_id,
        "snapshot_path": snapshot.get("path"),
        "snapshot_sha256": snapshot.get("sha256"),
        "snapshot_source_db_identity": snapshot.get("source_db_identity"),
        "execution_run_uuid": snapshot.get("execution_run_uuid"),
        "pe8_evidence": generation.manifest.get("pe8_evidence"),
        "disclosure": generation.manifest.get("disclosure"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6A fixed-15 manager lineup evaluation")
    parser.add_argument(
        "--gw", type=int, required=True,
        help="planning Gameweek.  Required: there is no default Gameweek.",
    )
    parser.add_argument(
        "--generation", default=None,
        help="explicit certified generation id SELECTOR.  Omit to resolve the current_generation "
             "pointer for this event.  It selects WHICH certified world to describe; the evidence "
             "that authorises the load is the persisted generation itself",
    )
    parser.add_argument("--config")
    parser.add_argument("--minutes-run", type=int, help="must equal the certified minutes run")
    parser.add_argument("--xpts-run", type=int, help="must equal the certified xPts run")
    parser.add_argument("--team-run", type=int, help="must equal the certified team-strength run")
    parser.add_argument(
        "--monte-carlo-run", type=int,
        help="must equal the certified Monte Carlo run recorded for this event, when it records one",
    )
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out-dir")
    parser.add_argument("--cutoff", help="planning as-of cutoff for coherent pre-deadline provenance")
    args = parser.parse_args(argv)

    planning_event = int(args.gw)
    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        # The certified GENERATION is resolved FIRST: it is the evidence every
        # predictive read below consumes, and the authoritative source of the exact
        # frozen run ids and of the pinned snapshot identity.
        try:
            # This is a MANAGER-WORLD packet: the certified generation whose horizon
            # kind corresponds to it is the one it describes, and the pointer it
            # resolves is that kind's.  A four-Gameweek transfer generation is a
            # different certified world and is never substituted for it.
            generation = gs.resolve_generation(
                conn, planning_event=planning_event, generation_id=args.generation,
                horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD,
            )
            gs.require_snapshot_retained(generation)
            gs.assert_generation_bundles_valid(conn, generation)
        except (gs.GenerationRefused, certified_bundle.CertificationRefused) as failure:
            print(f"phase 6a refused: {failure}", file=sys.stderr)
            return 3
        try:
            runs = certified_run_ids(
                generation, event=planning_event, minutes_run=args.minutes_run,
                xpts_run=args.xpts_run, team_run=args.team_run,
                monte_carlo_run=args.monte_carlo_run,
            )
        except (CertificationMismatch, certified_bundle.CertificationRefused) as failure:
            print(f"phase 6a refused: {failure}", file=sys.stderr)
            return 3
        versions = certified_model_versions(generation, event=planning_event)
        # The generation is resolved and re-proven BEFORE any squad work, so the
        # provenance this command publishes is printed first and an operator sees
        # WHICH certified world the packet describes even if a later step refuses.
        snapshot_provenance = certified_snapshot_provenance(generation)
        print(
            "  certified runs: "
            + " ".join(f"{family}={run_id}" for family, run_id in sorted(runs.items()))
            + f" generation={generation.generation_id[:24]}…"
            + f" snapshot={str(generation.snapshot.get('sha256'))[:16]}…"
        )
        print(
            "  generation provenance: "
            + json.dumps(
                {
                    "generation_id": snapshot_provenance["generation_id"],
                    "cutoff": generation.cutoff,
                    "snapshot_sha256": snapshot_provenance["snapshot_sha256"],
                    "pe8_consulted": (
                        snapshot_provenance.get("pe8_evidence") or {}
                    ).get("consulted"),
                },
                sort_keys=True,
            )
        )
        started = time.time()
        entry_id = config.get("fpl_entry_id")
        context = get_planning_context(
            conn, int(entry_id), int(planning_event), as_of=args.cutoff, season=config.get("season"),
            official_price_stale_after_hours=config.get("report", {}).get("official_price_stale_after_hours"),
        )
        squad = manager_worlds.resolve_squad(context, conn)
        if squad["player_count"] != 15:
            print(f"phase 6a failed: PlanningContext squad has {squad['player_count']} players, not 15", file=sys.stderr)
            return 3
        if not all(position in ("GKP", "DEF", "MID", "FWD") for position in squad["positions"].values()):
            print("phase 6a failed: unresolved player position in squad", file=sys.stderr)
            return 3

        timed = {}
        mark = time.time()
        built = manager_worlds.build_manager_worlds(
            conn, generation=generation, planning_event=int(planning_event),
            squad_ids=squad["squad_ids"],
            simulations=int(args.simulations), seed=int(args.seed), occupancy_audit=True,
        )
        world_matrix = built["world_matrix"]
        timed["manager_world_generation_s"] = round(time.time() - mark, 2)

        mark = time.time()
        skeletons = list(manager_lineup.enumerate_skeletons(squad["squad_ids"], squad["positions"]))
        timed["enumeration_s"] = round(time.time() - mark, 3)

        mark = time.time()
        ranked = manager_lineup.rank_policies(
            squad["squad_ids"], squad["positions"], world_matrix, top_k=int(args.top_k)
        )
        timed["ranking_s"] = round(time.time() - mark, 2)
        top_policies = ranked["top_policies"]

        mark = time.time()
        top_rows = []
        for policy in top_policies:
            metrics = manager_lineup.evaluate_policy(policy, world_matrix, squad["positions"])
            row = {
                **policy.as_dict(squad["names"]),
                **{key: value for key, value in metrics.items() if key != "final_counted_player_probability"},
                "mean_core_rank_only": ranked["top_mean_by_key"].get(policy.ordering_key()),
            }
            top_rows.append(row)
        top_rows.sort(key=lambda row: -row["mean_core"])
        timed["top_k_distribution_s"] = round(time.time() - mark, 2)
        timed["total_s"] = round(time.time() - started, 1)

        manager_section = {
            "phase_version": PHASE6A_VERSION,
            "lineup_model_version": manager_lineup.MANAGER_LINEUP_VERSION,
            "worlds_model_version": manager_worlds.MANAGER_WORLDS_VERSION,
            "planning_context_hash": _context_hash(context),
            # PE-9: the packet describes the CERTIFIED generation it consumed -- the
            # generation id, the exact certified run ids, the versions the manifest
            # records, the pinned snapshot identity and the PE-8 evidence references --
            # instead of restating CLI defaults.
            "generation_id": generation.generation_id,
            "generation_horizon_kind": generation.horizon_kind,
            "generation_cutoff": generation.cutoff,
            "generation_events": list(generation.events),
            "generation_snapshot": snapshot_provenance,
            "predictive_runs": {
                "minutes": int(runs["minutes_v1"]),
                "minutes_version": versions.get("minutes_v1"),
                "xpts": int(runs["xpts_v1"]),
                "xpts_version": versions.get("xpts_v1"),
                "team": int(runs["team_strength_v1"]),
                "team_version": versions.get("team_strength_v1"),
                "monte_carlo": (
                    int(runs["monte_carlo_v1"]) if "monte_carlo_v1" in runs else None
                ),
                "monte_carlo_version": built["simulation"]["mc_model_version"],
            },
            "simulation": built["simulation"],
            "input_run_ids": built["input_run_ids"],
            "squad_ids": squad["squad_ids"],
            "squad_state": squad["squad_state"],
            "squad_source": squad["squad_source"],
            "skeleton_count": ranked["skeletons"],
            "evaluated_policy_count": ranked["evaluated_policies"],
            "timing": timed,
            "scoring_basis": "CORE",
            "bonus_handling_status": "BONUS_MEAN_MANAGER_INTEGRATION_DEFERRED",
            "risk_flags": [
                "CORE_BASED_NO_BONUS_VARIANCE",
                "MANAGER_LAYER_FIXED_15_ONLY",
                "DESCRIPTIVE_ONLY_NO_EXECUTION",
            ],
            "top_policies": top_rows[:20],
        }
        built_packet = packet_mod.build_decision_packet(
            conn, config, event=int(planning_event), as_of=args.cutoff, manager=manager_section
        )
        packet = built_packet["packet"]

        out_dir = Path(args.out_dir) if args.out_dir else config_path(config, "exports_dir") / "manager"
        target = out_dir / f"gw{int(planning_event):02d}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "manager_lineup_packet.json").write_text(
            packet_mod.packet_to_json(packet), encoding="utf-8"
        )
        (target / "manager_lineup_packet.md").write_text(
            packet_mod.render_packet_markdown(packet), encoding="utf-8"
        )
        elapsed = time.time() - started
        print(
            f"phase 6a complete: GW{planning_event} squad={len(squad['squad_ids'])} "
            f"worlds={built['simulation']['worlds']} skeletons={ranked['skeletons']} "
            f"policies={ranked['evaluated_policies']} occ_viol={built['simulation']['occupancy_violations']} "
            f"elapsed={elapsed:.1f}s artifact={target}"
        )
        pd = None
        try:
            pd = packet_mod.packet_sha256(packet)
        except Exception:  # pragma: no cover - defensive
            pd = None
        print(f"  manager packet digest: {pd}")
        for index, row in enumerate(top_rows[:5], start=1):
            print(
                f"  #{index} mean={row['mean_core']:.3f} cap={row['captain_name']} vice={row['vice_captain_name']} "
                f"XI={row['starter_names']} bench={row['bench_names']}"
            )
        return 0
    finally:
        conn.close()


def _context_hash(context) -> str:
    from fpl_brain import analytics
    return analytics.canonical_hash({
        "entry_id": context.entry_id, "event": context.planning_event,
        "squad": context.squad, "official_runs": context.official_runs,
    })


if __name__ == "__main__":
    raise SystemExit(main())
