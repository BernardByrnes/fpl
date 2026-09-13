"""Certified predictive bundle / DAG contract (R4A).

Horizon support as previously implemented proved only: same planning event, same
family, same cutoff, ``complete``.  That is insufficient, because predictive
families form a dependency graph:

    minutes ─┐
    team    ─┼─► xpts ─┐
    rates   ─┘         ├─► monte_carlo
                       ┘

A same-cutoff mix of families that do not actually reference one another is not a
certifiable bundle, and a newer same-cutoff rerun of one family must not silently
replace the certified one.

This module makes the bundle explicit: exact run ids per family per event, plus
validation of the dependency edges, the data/code snapshot identity, the model
versions and the planning context.  The decision engine consumes the bundle's
EXACT run ids; it never rediscoveres "latest matching run" per family.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

DIAG_PREDICTIVE_BUNDLE_INCOHERENT = "PREDICTIVE_BUNDLE_INCOHERENT"

# family -> upstream families its run must reference
BUNDLE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "minutes_v1": (),
    "team_strength_v1": (),
    "player_rates_v1": (),
    "xpts_v1": ("minutes_v1", "team_strength_v1", "player_rates_v1"),
    "monte_carlo_v1": ("minutes_v1", "team_strength_v1", "player_rates_v1", "xpts_v1"),
}

# family -> (columns on its own table carrying the upstream run ids, source table)
_XPTS_UPSTREAM = {
    "minutes_v1": "minutes_run_id",
    "team_strength_v1": "team_run_id",
    "player_rates_v1": "rate_run_id",
}

MC_UPSTREAM = {
    "minutes_v1": "minutes_run_id",
    "team_strength_v1": "team_run_id",
    "player_rates_v1": "rate_run_id",
    "xpts_v1": "xpts_run_id",
}


class BundleIncoherent(RuntimeError):
    """The candidate families do not form one coherent predictive bundle."""

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(f"{DIAG_PREDICTIVE_BUNDLE_INCOHERENT}: " + "; ".join(reasons))
        self.reasons = list(reasons)


@dataclass(frozen=True)
class CertifiedBundle:
    """Exact run ids and identities for one event's certified predictions."""

    event: int
    cutoff: str
    runs: Mapping[str, int]
    model_versions: Mapping[str, str]
    code_snapshot_sha256: str | None = None
    data_snapshot_sha256: str | None = None
    planning_context_hash: str | None = None
    required_versions: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event),
            "cutoff": self.cutoff,
            "runs": {k: int(v) for k, v in self.runs.items()},
            "model_versions": dict(self.model_versions),
            "code_snapshot_sha256": self.code_snapshot_sha256,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "planning_context_hash": self.planning_context_hash,
        }

    def bundle_identity(self) -> str:
        """Deterministic identity of this bundle (exact ids + identities)."""

        return "sha256:" + hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()


def _run(conn: sqlite3.Connection, run_id: int) -> Mapping[str, Any] | None:
    return conn.execute("SELECT * FROM projection_runs WHERE id=?", (int(run_id),)).fetchone()


def _upstream_run_ids(conn: sqlite3.Connection, family: str, run_id: int) -> dict[str, int | None]:
    """Upstream run ids a family's rows actually reference, if recorded."""

    if family == "xpts_v1":
        row = conn.execute(
            "SELECT minutes_run_id, team_run_id, rate_run_id FROM player_fixture_xpts_projections"
            " WHERE projection_run_id=? LIMIT 1",
            (int(run_id),),
        ).fetchone()
        if row is None:
            return {}
        return {
            "minutes_v1": row["minutes_run_id"],
            "team_strength_v1": row["team_run_id"],
            "player_rates_v1": row["rate_run_id"],
        }
    if family == "monte_carlo_v1":
        row = conn.execute(
            "SELECT minutes_run_id, team_run_id, rate_run_id, xpts_run_id"
            " FROM monte_carlo_distributions WHERE projection_run_id=? LIMIT 1",
            (int(run_id),),
        ).fetchone()
        if row is None:
            return {}
        return {
            "minutes_v1": row["minutes_run_id"],
            "team_strength_v1": row["team_run_id"],
            "player_rates_v1": row["rate_run_id"],
            "xpts_v1": row["xpts_run_id"],
        }
    return {}


def validate_certified_bundle(
    conn: sqlite3.Connection,
    *,
    event: int,
    cutoff: str,
    runs: Mapping[str, int],
    required_versions: Mapping[str, str] | None = None,
    data_snapshot_sha256: str | None = None,
    code_snapshot_sha256: str | None = None,
) -> CertifiedBundle:
    """Validate one event's candidate families as a coherent predictive bundle.

    Raises :class:`BundleIncoherent` with every reason found, so a caller can
    report all incoherences at once instead of the first.
    """

    reasons: list[str] = []
    versions: dict[str, str] = {}
    context_hashes: set[str] = set()
    code_snapshots: set[str] = set()
    cutoffs: set[str] = set()

    for family in BUNDLE_DEPENDENCIES:
        run_id = runs.get(family)
        if run_id is None:
            reasons.append(f"missing family {family}")
            continue
        row = _run(conn, int(run_id))
        if row is None:
            reasons.append(f"run {run_id} for {family} does not exist")
            continue
        if str(row["model_family"]) != family:
            reasons.append(
                f"run {run_id} is family {row['model_family']!r}, expected {family!r}"
            )
        if int(row["planning_event"]) != int(event):
            reasons.append(
                f"run {run_id} ({family}) is planning_event {row['planning_event']}, expected {event}"
            )
        if str(row["status"]) != "complete":
            reasons.append(f"run {run_id} ({family}) status is {row['status']!r}, not complete")
        if str(row["data_cutoff"]) != str(cutoff):
            reasons.append(
                f"run {run_id} ({family}) data_cutoff {row['data_cutoff']} != bundle cutoff {cutoff}"
            )
        cutoffs.add(str(row["data_cutoff"]))
        versions[family] = str(row["model_version"])
        if row["planning_context_hash"]:
            context_hashes.add(str(row["planning_context_hash"]))
        # projection_runs.source_snapshot_sha256 is the CODE fingerprint, not a data
        # identity; it is checked for cross-family consistency and against an
        # explicitly supplied code_snapshot_sha256, never against a DATA snapshot.
        if row["source_snapshot_sha256"]:
            code_snapshots.add(str(row["source_snapshot_sha256"]))

    # Dependency edges: xPts and Monte Carlo must reference THIS bundle's runs.
    for family in ("xpts_v1", "monte_carlo_v1"):
        run_id = runs.get(family)
        if run_id is None:
            continue
        upstream = _upstream_run_ids(conn, family, int(run_id))
        if not upstream:
            reasons.append(
                f"{family} run {run_id} records no upstream run ids; its dependency edges cannot be verified"
            )
            continue
        for dep_family, column in (
            _XPTS_UPSTREAM.items() if family == "xpts_v1" else MC_UPSTREAM.items()
        ):
            expected = runs.get(dep_family)
            actual = upstream.get(dep_family)
            if expected is None:
                continue
            if actual is None:
                reasons.append(f"{family} run {run_id} has no {column}; expected {expected}")
            elif int(actual) != int(expected):
                reasons.append(
                    f"{family} run {run_id} references {dep_family} run {actual}, "
                    f"but the bundle declares {expected}"
                )

    if required_versions:
        for family, wanted in required_versions.items():
            got = versions.get(family)
            if got is not None and got != wanted:
                reasons.append(f"{family} model_version {got!r} != required {wanted!r}")

    if len(context_hashes) > 1:
        reasons.append(f"planning_context_hash differs across families: {sorted(context_hashes)}")

    if code_snapshot_sha256 is not None and code_snapshots:
        if code_snapshots != {str(code_snapshot_sha256)}:
            reasons.append(
                f"family code snapshots {sorted(code_snapshots)} != bundle code snapshot "
                f"{code_snapshot_sha256}"
            )
    elif len(code_snapshots) > 1:
        reasons.append(
            f"code snapshot differs across families (all must come from one code revision): "
            f"{sorted(code_snapshots)}"
        )
    # A DATA snapshot identity is recorded by the certification artifact (and carried
    # on the bundle for the decision engine); it is deliberately not compared against
    # the code fingerprint column.

    if reasons:
        raise BundleIncoherent(reasons)

    return CertifiedBundle(
        event=int(event),
        cutoff=str(cutoff),
        runs={k: int(v) for k, v in runs.items() if k in BUNDLE_DEPENDENCIES},
        model_versions=versions,
        code_snapshot_sha256=code_snapshot_sha256,
        data_snapshot_sha256=data_snapshot_sha256,
        planning_context_hash=next(iter(context_hashes)) if len(context_hashes) == 1 else None,
        required_versions=dict(required_versions or {}),
    )


def certified_bundle_from_explicit_ids(
    conn: sqlite3.Connection,
    *,
    event: int,
    cutoff: str,
    runs: Mapping[str, int],
    data_snapshot_sha256: str | None = None,
    code_snapshot_sha256: str | None = None,
    required_versions: Mapping[str, str] | None = None,
) -> CertifiedBundle:
    """Certify an explicitly supplied bundle (the decision engine's entry point).

    There is deliberately no "discover the latest run per family" helper here:
    silent rediscovery is exactly the failure mode this module exists to prevent.
    """

    return validate_certified_bundle(
        conn,
        event=int(event),
        cutoff=str(cutoff),
        runs=runs,
        required_versions=required_versions,
        data_snapshot_sha256=data_snapshot_sha256,
        code_snapshot_sha256=code_snapshot_sha256,
    )


def certify_horizon_bundles(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    cutoff: str,
    runs_by_event: Mapping[int, Mapping[str, int]],
    data_snapshot_sha256: str | None = None,
    required_versions: Mapping[str, str] | None = None,
) -> dict[int, CertifiedBundle]:
    """Certify every event's bundle, raising on the first incoherent event."""

    certified: dict[int, CertifiedBundle] = {}
    failures: dict[int, list[str]] = {}
    for event in events:
        try:
            certified[int(event)] = certified_bundle_from_explicit_ids(
                conn,
                event=int(event),
                cutoff=cutoff,
                runs=runs_by_event[int(event)],
                data_snapshot_sha256=data_snapshot_sha256,
                required_versions=required_versions,
            )
        except BundleIncoherent as exc:
            failures[int(event)] = exc.reasons
    if failures:
        flat = [f"GW{event}: {reason}" for event, reasons in sorted(failures.items()) for reason in reasons]
        raise BundleIncoherent(flat)
    return certified


__all__ = [
    "BUNDLE_DEPENDENCIES",
    "BundleIncoherent",
    "CertifiedBundle",
    "DIAG_PREDICTIVE_BUNDLE_INCOHERENT",
    "certified_bundle_from_explicit_ids",
    "certify_horizon_bundles",
    "validate_certified_bundle",
]
