"""Deterministic walk-forward evaluation population and comparison identity.

WHAT THIS IS
------------
The core answers one question before any metric is computed:

    Exactly which player/event observations are eligible to be scored, and is
    every model/baseline arm being compared on exactly the same population?

It builds the population.  It does not compute a scoreboard, it does not fetch
anything, it does not generate predictions, and it does not mutate history.  It
consumes persisted predictions and persisted outcomes.

TWO GRAINS, EXPLICIT AND SEPARATE
---------------------------------
* ``player_event`` is the headline grain, and the ONLY grain a points baseline
  may be compared on.  The xPts model is fixture-grain and is AGGREGATED UP
  (summed over that player's fixtures in the event); the points baselines are
  event-grain and stay there.  Expanding an event-grain baseline onto every DGW
  fixture is forbidden: it would invent a per-fixture claim the baseline never
  made and would silently double-count it.
* ``player_fixture`` is the diagnostic grain for components whose target
  naturally exists per fixture (minutes, goals, assists, saves, clean sheet,
  DefCon).  Those rows never enter the headline points population.

Digests are namespaced by grain, so a fixture-grain digest can never be
mistaken for an event-grain one.

MISSING IS NEVER ZERO
---------------------
Every candidate gets an explicit status.  ``MODEL_PROJECTION_MISSING`` is not
zero points, ``OUTCOME_PLACEHOLDER_EXCLUDED`` is not a non-appearance, and a
blank Gameweek is not a model failure.  Excluded candidates are returned and
counted, never dropped, so coverage is visible instead of implied.

SAME-POPULATION GATE
--------------------
A model-vs-baseline comparison is valid only when both arms cover the identical
key set.  :func:`assert_same_population` fails closed and names the keys that
differ, so two MAE numbers computed on different rows cannot be reported as a
comparison.

DETERMINISM
-----------
Given the same certified bundle, outcome state, policy version and target
events, the rows, the status counts, the digest and the evaluation identity are
byte-for-byte reproducible.  Nothing depends on database row order or on the
current time.

PROVENANCE
----------
Eligibility is decided from PERSISTED identity -- each run's ``model_version``
against the live accepted constant for its family, its ``data_cutoff`` against
the bundle cutoff, and its ``source_snapshot_sha256`` against the anchor's code
identity -- never from a display string alone.  Pre-PE1 runs therefore cannot
enter a corrected scoreboard because their names happen to look compatible.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import analytics
from . import execution
from . import minutes_model
from . import player_rates
from . import repositories as repo
from . import substitution_model
from . import team_model
from . import monte_carlo
from . import xpts as xpts_module

WALK_FORWARD_VERSION = "walk_forward_v1.0.0"
MISSING_DATA_POLICY_VERSION = "wf_missing_policy_v1.0.0"

GRAIN_PLAYER_EVENT = "player_event"
GRAIN_PLAYER_FIXTURE = "player_fixture"

# -- candidate statuses -----------------------------------------------------
EVALUATED = "EVALUATED"
TARGET_NO_FIXTURE = "TARGET_NO_FIXTURE"
TARGET_FIXTURE_NOT_PLAYED = "TARGET_FIXTURE_NOT_PLAYED"
OUTCOME_NOT_FINALISED = "OUTCOME_NOT_FINALISED"
OUTCOME_PLACEHOLDER_EXCLUDED = "OUTCOME_PLACEHOLDER_EXCLUDED"
PLAYER_NOT_IN_OFFICIAL_POOL = "PLAYER_NOT_IN_OFFICIAL_POOL"
MODEL_PROJECTION_MISSING = "MODEL_PROJECTION_MISSING"
INPUT_EVIDENCE_UNAVAILABLE = "INPUT_EVIDENCE_UNAVAILABLE"

CANDIDATE_STATUSES = (
    EVALUATED,
    TARGET_NO_FIXTURE,
    TARGET_FIXTURE_NOT_PLAYED,
    OUTCOME_NOT_FINALISED,
    OUTCOME_PLACEHOLDER_EXCLUDED,
    PLAYER_NOT_IN_OFFICIAL_POOL,
    MODEL_PROJECTION_MISSING,
    INPUT_EVIDENCE_UNAVAILABLE,
)

#: Baseline families whose payload is a claim about FPL POINTS, and which may
#: therefore appear in the headline points comparison.
HEADLINE_POINTS_BASELINE_KINDS = (
    analytics.EP_NEXT_KIND,
    analytics.RECENT_POINTS_KIND,
    analytics.NAIVE_P90_KIND,
)

#: Baselines that are real baselines but do NOT predict FPL points.  Comparing
#: minutes against points would be a category error, so they are excluded from
#: the points score and belong to component diagnostics.
NON_POINTS_BASELINE_KINDS = (analytics.NAIVE_MINUTES_KIND,)

#: The accepted current version for each family the anchor must provide.  These
#: are the live constants, not display strings copied from a payload.
CURRENT_FAMILY_VERSIONS: dict[str, str] = {
    "minutes_v1": minutes_model.MINUTES_MODEL_VERSION,
    "team_strength_v1": team_model.TEAM_MODEL_VERSION,
    "team_baseline": team_model.TEAM_BASELINE_MODEL_VERSION,
    "baseline": analytics.BASELINE_MODEL_VERSION,
    "player_rates_v1": player_rates.PLAYER_RATE_MODEL_VERSION,
    "player_rate_baseline": player_rates.PLAYER_RATE_BASELINE_MODEL_VERSION,
    "xpts_v1": xpts_module.XPTS_MODEL_VERSION,
    "monte_carlo_v1": monte_carlo.MONTE_CARLO_MODEL_VERSION,
}


class WalkForwardError(RuntimeError):
    """The evaluation could not be constructed from the available identity."""


class PopulationMismatch(WalkForwardError):
    """Two arms were about to be compared on different eligible populations."""

    def __init__(self, message: str, *, model_only: Sequence[Any], baseline_only: Sequence[Any]) -> None:
        super().__init__(message)
        self.model_only = tuple(model_only)
        self.baseline_only = tuple(baseline_only)


# ---------------------------------------------------------------------------
# Canonical identity
# ---------------------------------------------------------------------------


def canonical_population_digest(keys: Iterable[Sequence[Any]], *, grain: str) -> str:
    """SHA-256 over the canonically serialised eligible key set.

    Canonical serialisation is the grain namespace, then one
    ``'|'``-joined key per line in sorted order, newline-terminated.  Sorting
    makes it independent of database row order; the namespace makes an
    event-grain digest unrepresentable as a fixture-grain one.
    """

    if grain not in (GRAIN_PLAYER_EVENT, GRAIN_PLAYER_FIXTURE):
        raise WalkForwardError(f"unknown grain {grain!r}")
    lines = sorted("|".join(str(int(part)) for part in key) for key in keys)
    payload = f"{WALK_FORWARD_VERSION}\n{grain}\n" + "\n".join(lines)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluationIdentity:
    """Enough information to reproduce the population that was judged."""

    evaluation_version: str
    grain: str
    target_events: tuple[int, ...]
    planning_cutoff: str | None
    projection_run_ids: tuple[tuple[str, int], ...]
    baseline_run_ids: tuple[tuple[str, int], ...]
    baseline_kinds: tuple[str, ...]
    eligible_population_digest: str
    model_versions: tuple[tuple[str, str], ...]
    code_snapshot_sha256: str | None
    code_revision: str | None
    missing_data_policy_version: str
    outcome_state: tuple[tuple[int, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_version": self.evaluation_version,
            "grain": self.grain,
            "target_events": [int(e) for e in self.target_events],
            "planning_cutoff": self.planning_cutoff,
            "projection_run_ids": {name: int(rid) for name, rid in self.projection_run_ids},
            "baseline_run_ids": {name: int(rid) for name, rid in self.baseline_run_ids},
            "baseline_kinds": [str(k) for k in self.baseline_kinds],
            "eligible_population_digest": self.eligible_population_digest,
            "model_versions": {name: str(v) for name, v in self.model_versions},
            "code_snapshot_sha256": self.code_snapshot_sha256,
            "code_revision": self.code_revision,
            "missing_data_policy_version": self.missing_data_policy_version,
            "outcome_state": {int(e): str(s) for e, s in self.outcome_state},
        }


# ---------------------------------------------------------------------------
# Anchor discovery: certified identity, never "the newest rows"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorEvent:
    event: int
    cutoff: str
    code_snapshot_sha256: str | None
    model_runs: tuple[tuple[str, int], ...]
    model_versions: tuple[tuple[str, str], ...]
    outcome_state: str

    def run_id(self, family: str) -> int | None:
        for name, run in self.model_runs:
            if name == family:
                return run
        return None


@dataclass(frozen=True)
class CertifiedAnchor:
    certification_identity: str | None
    planning_cutoff: str | None
    events: tuple[AnchorEvent, ...]

    @property
    def event_ids(self) -> tuple[int, ...]:
        return tuple(entry.event for entry in self.events)

    def for_event(self, event: int) -> AnchorEvent:
        for entry in self.events:
            if entry.event == int(event):
                return entry
        raise WalkForwardError(f"event {int(event)} is not part of the certified anchor")


def _recorded_code_snapshot(conn: sqlite3.Connection, run_ids: Iterable[int]) -> str | None:
    """The code identity the anchor's own runs agree on.

    The bundle-level field may be absent, so the runs themselves are the
    authority -- the same aggregation ``certified_bundle`` performs.  Disagreement
    is reported rather than resolved by picking one.
    """

    seen = {
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT source_snapshot_sha256 FROM projection_runs "
            f"WHERE id IN ({','.join('?' for _ in list(run_ids))}) AND source_snapshot_sha256 IS NOT NULL",
            tuple(int(rid) for rid in run_ids),
        ).fetchall()
    }
    if not seen:
        return None
    if len(seen) > 1:
        raise WalkForwardError(f"the anchor's runs disagree on their code snapshot: {sorted(seen)}")
    return seen.pop()


def discover_certified_anchor(
    conn: sqlite3.Connection,
    artifact: Mapping[str, Any],
    *,
    events: Sequence[int] | None = None,
) -> CertifiedAnchor:
    """Resolve the certified bundle's runs into a deterministic anchor.

    Runs come from the certification artifact, not from a "newest row" query, so
    a later unrelated run cannot silently become the evaluation anchor.
    """

    bundles = artifact.get("certified_bundles") or {}
    if not bundles:
        raise WalkForwardError("the certification artifact declares no certified bundles")
    wanted = sorted(int(e) for e in (events if events is not None else bundles.keys()))
    entries: list[AnchorEvent] = []
    for event in wanted:
        bundle = bundles.get(str(event)) or bundles.get(event)
        if not bundle:
            raise WalkForwardError(f"the certification artifact has no bundle for event {event}")
        runs = tuple(sorted((str(name), int(rid)) for name, rid in (bundle.get("runs") or {}).items()))
        if not runs:
            raise WalkForwardError(f"event {event} declares no runs")
        cutoff = str(bundle.get("cutoff") or artifact.get("planning_cutoff") or "").strip()
        if not cutoff:
            raise WalkForwardError(f"event {event} declares no cutoff")
        entries.append(
            AnchorEvent(
                event=event,
                cutoff=cutoff,
                code_snapshot_sha256=_recorded_code_snapshot(conn, [rid for _name, rid in runs]),
                model_runs=runs,
                model_versions=tuple(sorted((str(k), str(v)) for k, v in (bundle.get("model_versions") or {}).items())),
                outcome_state=_outcome_state(conn, event),
            )
        )
    return CertifiedAnchor(
        certification_identity=str(artifact.get("four_gw_certification_identity") or "") or None,
        planning_cutoff=str(artifact.get("planning_cutoff") or "").strip() or None,
        events=tuple(entries),
    )


def _outcome_state(conn: sqlite3.Connection, event: int) -> str:
    from . import planning

    state, _reasons = planning.event_data_state(conn, int(event))
    return str(state)


# ---------------------------------------------------------------------------
# Semantic eligibility
# ---------------------------------------------------------------------------


def run_is_current_semantics(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    expected_cutoff: str,
    expected_code_snapshot: str | None,
) -> tuple[bool, tuple[str, ...]]:
    """Whether a persisted run may enter a CORRECTED scoreboard.

    Decided from persisted provenance: the family's accepted version constant,
    the run's data cutoff, and its code fingerprint.  A pre-repair run fails on
    at least one of those even when its model name looks compatible.
    """

    row = conn.execute(
        "SELECT model_family, model_version, data_cutoff, source_snapshot_sha256 "
        "FROM projection_runs WHERE id=?",
        (int(run_id),),
    ).fetchone()
    if row is None:
        return False, (f"run {int(run_id)} does not exist",)
    values = dict(row) if isinstance(row, sqlite3.Row) else dict(zip(
        ("model_family", "model_version", "data_cutoff", "source_snapshot_sha256"), row))
    reasons: list[str] = []
    family = str(values["model_family"])
    accepted = CURRENT_FAMILY_VERSIONS.get(family)
    if accepted is not None and str(values["model_version"]) != accepted:
        reasons.append(f"{family} version {values['model_version']!r} != accepted {accepted!r}")
    if str(values["data_cutoff"]) != str(expected_cutoff):
        reasons.append(f"data cutoff {values['data_cutoff']!r} != anchor cutoff {expected_cutoff!r}")
    recorded = values["source_snapshot_sha256"]
    if expected_code_snapshot is not None:
        if not recorded:
            reasons.append("run records no code fingerprint")
        elif str(recorded) != str(expected_code_snapshot):
            reasons.append(
                f"code fingerprint {str(recorded)[:12]} != anchor {str(expected_code_snapshot)[:12]}"
            )
    return (not reasons), tuple(reasons)


# ---------------------------------------------------------------------------
# Model aggregation and baseline resolution
# ---------------------------------------------------------------------------


def model_event_xpts(conn: sqlite3.Connection, run_id: int) -> dict[int, dict[str, Any]]:
    """Aggregate a run's fixture xPts to player x event.

    ``xpts`` is the SUM over that player's fixture rows in the event: one
    fixture for a single gameweek, two for a double, none for a blank.  The
    contributing fixture ids are retained so the grain is never implicit.
    """

    totals: dict[int, dict[str, Any]] = {}
    for record in analytics.xpts_projections(conn, int(run_id)):
        player_id = int(record["player_id"])
        event = int(record["event"])
        key = (event, player_id)
        value = (record.get("payload") or {}).get("total_xpts")
        if value is None:
            continue
        entry = totals.setdefault(key, {"event": event, "player_id": player_id, "xpts": 0.0, "fixtures": []})
        entry["xpts"] += float(value)
        entry["fixtures"].append(int(record["fixture_id"]))
    for entry in totals.values():
        entry["fixtures"].sort()
    return totals


def resolve_baseline_run(
    conn: sqlite3.Connection,
    anchor_event: AnchorEvent,
    *,
    model_family: str = analytics.BASELINE_MODEL_FAMILY,
) -> int | None:
    """The baseline run belonging to THIS anchor, by family + event + cutoff + code.

    Uses the repository's read-only completed-run lookup so the anchor's own
    run is selected on identity, not on recency.
    """

    return execution.find_completed_projection_run(
        conn,
        model_family=str(model_family),
        planning_event=int(anchor_event.event),
        data_cutoff=str(anchor_event.cutoff),
        source_snapshot_sha256=anchor_event.code_snapshot_sha256,
    )


def baseline_event_values(conn: sqlite3.Connection, run_id: int, kind: str) -> dict[tuple[int, int], float]:
    """Event-grain baseline points, keyed ``(event, player_id)``.

    One value per player per event.  This mapping is never expanded across
    fixtures.
    """

    values: dict[tuple[int, int], float] = {}
    for record in analytics.frozen_predictions(conn, int(run_id), [kind]):
        value = (record.get("payload") or {}).get("value")
        if value is None:
            continue
        values[(int(record["event"]), int(record["player_id"]))] = float(value)
    return values


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


@dataclass
class Population:
    grain: str
    baseline_kind: str | None
    rows: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    digest: str = ""
    identity: EvaluationIdentity | None = None

    @property
    def eligible_keys(self) -> list[tuple[int, int]]:
        return [(int(r["event"]), int(r["player_id"])) for r in self.rows]

    def coverage(self) -> dict[str, Any]:
        total = len(self.rows) + len(self.excluded)
        return {
            "candidates": total,
            "evaluated": len(self.rows),
            "excluded": len(self.excluded),
            "by_status": dict(self.status_counts),
            "evaluated_share": (len(self.rows) / total) if total else None,
        }


def assert_same_population(
    model_keys: Iterable[Sequence[Any]], baseline_keys: Iterable[Sequence[Any]]
) -> None:
    """Fail closed when two arms do not cover the identical eligible key set."""

    model_set = {tuple(int(p) for p in key) for key in model_keys}
    baseline_set = {tuple(int(p) for p in key) for key in baseline_keys}
    if model_set == baseline_set:
        return
    model_only = sorted(model_set - baseline_set)
    baseline_only = sorted(baseline_set - model_set)
    raise PopulationMismatch(
        f"the arms cover different populations: {len(model_only)} model-only key(s), "
        f"{len(baseline_only)} baseline-only key(s); refusing to compare them",
        model_only=model_only,
        baseline_only=baseline_only,
    )


def build_event_population(
    conn: sqlite3.Connection,
    *,
    anchor: CertifiedAnchor,
    events: Sequence[int],
    baseline_kind: str | None = None,
    require_same_population: bool = True,
) -> Population:
    """The headline player×event population for the requested events.

    Every candidate the official pool offers for a target event is classified;
    nothing is dropped silently and nothing missing becomes zero.
    """

    population = Population(grain=GRAIN_PLAYER_EVENT, baseline_kind=baseline_kind)
    projection_runs: list[tuple[str, int]] = []
    baseline_runs: list[tuple[str, int]] = []
    outcome_state: list[tuple[int, str]] = []
    versions: list[tuple[str, str]] = []
    code_snapshot: str | None = None
    cutoff: str | None = None

    for event in sorted(int(e) for e in events):
        entry = anchor.for_event(event)
        cutoff = cutoff or entry.cutoff
        code_snapshot = code_snapshot or entry.code_snapshot_sha256
        xpts_run = entry.run_id("xpts_v1")
        if xpts_run is None:
            raise WalkForwardError(f"the anchor for event {event} declares no xpts run")
        current, reasons = run_is_current_semantics(
            conn, xpts_run, expected_cutoff=entry.cutoff, expected_code_snapshot=entry.code_snapshot_sha256
        )
        if not current:
            raise WalkForwardError(f"xpts run {xpts_run} is not current semantics: {'; '.join(reasons)}")
        projection_runs.append(("xpts_v1", xpts_run))
        versions.append(("xpts_v1", xpts_run and _run_version(conn, xpts_run)))

        state = entry.outcome_state
        outcome_state.append((event, state))
        model = model_event_xpts(conn, xpts_run)
        pool = _official_pool(conn)
        fixtures_by_team = _fixtures_by_team(conn, event)
        played = _played_fixtures(conn, event)
        placeholder_pairs = _placeholder_pairs(conn, event)
        observed_pairs = _observed_pairs(conn, event)

        baseline_values: dict[tuple[int, int], float] = {}
        baseline_run = resolve_baseline_run(conn, entry) if baseline_kind else None
        if baseline_run is not None:
            baseline_runs.append((baseline_kind or "baseline", baseline_run))
            baseline_values = baseline_event_values(conn, baseline_run, str(baseline_kind))

        for player_id in sorted(pool):
            key = (event, int(player_id))
            team_id = pool[int(player_id)]
            fixtures = fixtures_by_team.get(int(team_id), [])
            candidate = {
                "event": event,
                "player_id": int(player_id),
                "team_id": int(team_id),
                "fixtures": sorted(fixtures),
            }
            status, reason = _classify(
                state=state,
                fixtures=fixtures,
                played=played,
                model=model,
                key=key,
                placeholder_pairs=placeholder_pairs,
                observed_pairs=observed_pairs,
            )
            if status != EVALUATED:
                candidate["status"] = status
                candidate["reason"] = reason
                population.excluded.append(candidate)
                population.status_counts[status] = population.status_counts.get(status, 0) + 1
                continue
            model_entry = model[key]
            candidate["status"] = EVALUATED
            candidate["model_xpts"] = round(float(model_entry["xpts"]), 6)
            candidate["model_fixtures"] = list(model_entry["fixtures"])
            candidate["outcome"] = outcome_event_totals(conn, key)
            if baseline_kind is not None:
                # Event-grain value, taken once.  Absent is absent: it is never 0.
                candidate["baseline_value"] = baseline_values.get(key)
            population.rows.append(candidate)
            population.status_counts[EVALUATED] = population.status_counts.get(EVALUATED, 0) + 1

    if baseline_kind is not None and require_same_population:
        model_keys = [key for key in population.eligible_keys if _row_has_model(population, key)]
        baseline_keys = [
            (int(r["event"]), int(r["player_id"]))
            for r in population.rows
            if r.get("baseline_value") is not None
        ]
        assert_same_population(model_keys, baseline_keys)

    population.rows.sort(key=lambda r: (int(r["event"]), int(r["player_id"])))
    population.excluded.sort(key=lambda r: (int(r["event"]), int(r["player_id"]), str(r.get("status"))))
    population.digest = canonical_population_digest(population.eligible_keys, grain=GRAIN_PLAYER_EVENT)
    population.identity = EvaluationIdentity(
        evaluation_version=WALK_FORWARD_VERSION,
        grain=GRAIN_PLAYER_EVENT,
        target_events=tuple(sorted(int(e) for e in events)),
        planning_cutoff=cutoff or anchor.planning_cutoff,
        projection_run_ids=tuple(sorted(projection_runs)),
        baseline_run_ids=tuple(sorted(baseline_runs)),
        baseline_kinds=(str(baseline_kind),) if baseline_kind else (),
        eligible_population_digest=population.digest,
        model_versions=tuple(sorted(versions)),
        code_snapshot_sha256=code_snapshot,
        code_revision=analytics.code_revision(),
        missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
        outcome_state=tuple(sorted(outcome_state)),
    )
    return population


def _row_has_model(population: Population, key: tuple[int, int]) -> bool:
    return any(
        (int(r["event"]), int(r["player_id"])) == key and r.get("model_xpts") is not None
        for r in population.rows
    )


def _run_version(conn: sqlite3.Connection, run_id: int) -> str:
    row = conn.execute("SELECT model_version FROM projection_runs WHERE id=?", (int(run_id),)).fetchone()
    return str(row[0]) if row else ""


def _official_pool(conn: sqlite3.Connection) -> dict[int, int]:
    return {
        int(row[0]): int(row[1])
        for row in conn.execute(
            "SELECT id, team_id FROM players WHERE is_active=1 AND team_id IS NOT NULL ORDER BY id"
        )
    }


def _fixtures_by_team(conn: sqlite3.Connection, event: int) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for row in conn.execute(
        "SELECT id, team_h, team_a FROM fixtures WHERE event=? AND id > 0 ORDER BY id", (int(event),)
    ):
        grouped.setdefault(int(row[1]), []).append(int(row[0]))
        grouped.setdefault(int(row[2]), []).append(int(row[0]))
    return grouped


def _played_fixtures(conn: sqlite3.Connection, event: int) -> set[int]:
    return {
        int(row[0])
        for row in conn.execute(
            "SELECT id FROM fixtures WHERE event=? AND finished=1 AND started=1", (int(event),)
        )
    }


def _placeholder_pairs(conn: sqlite3.Connection, event: int) -> set[tuple[int, int, int]]:
    """Rows that are scheduled placeholders, by the frozen canonical predicate."""

    return {
        (int(row[0]), int(row[1]), int(row[2]))
        for row in conn.execute(
            "SELECT pg.event, pg.player_id, pg.fixture_id FROM player_gameweeks pg "
            f"WHERE pg.event=? AND ({repo.scheduled_placeholder_sql('pg')})",
            (int(event),),
        )
    }


def _observed_pairs(conn: sqlite3.Connection, event: int) -> set[tuple[int, int, int]]:
    return {
        (int(event), int(row[0]), int(row[1]))
        for row in conn.execute(
            "SELECT player_id, fixture_id FROM player_gameweeks WHERE event=? AND fixture_id > 0",
            (int(event),),
        )
    }


def outcome_event_totals(conn: sqlite3.Connection, key: tuple[int, int]) -> dict[str, Any]:
    """Realised points and minutes for one player's whole event.

    Summed across the event's fixtures, so a double gameweek is one event total
    rather than whichever fixture happened to be read last.
    """

    rows = conn.execute(
        "SELECT total_points, minutes FROM player_gameweeks "
        "WHERE event=? AND player_id=? AND fixture_id > 0",
        (int(key[0]), int(key[1])),
    ).fetchall()
    if not rows:
        return {"total_points": None, "minutes": None}
    points = [row[0] for row in rows if row[0] is not None]
    minutes = [row[1] for row in rows if row[1] is not None]
    return {
        "total_points": sum(int(value) for value in points) if points else None,
        "minutes": sum(int(value) for value in minutes) if minutes else None,
    }


def _classify(
    *,
    state: str,
    fixtures: Sequence[int],
    played: set[int],
    model: Mapping[tuple[int, int], Any],
    key: tuple[int, int],
    placeholder_pairs: set[tuple[int, int, int]],
    observed_pairs: set[tuple[int, int, int]],
) -> tuple[str, str]:
    """The single status decision, in explicit precedence order."""

    if str(state) != "FINAL":
        return OUTCOME_NOT_FINALISED, f"event {key[0]} is {state}"
    if not fixtures:
        return TARGET_NO_FIXTURE, f"team has no fixture in event {key[0]}"
    unplayed = [fid for fid in fixtures if int(fid) not in played]
    if unplayed:
        return TARGET_FIXTURE_NOT_PLAYED, f"fixture(s) not played: {sorted(unplayed)}"
    if key not in model:
        return MODEL_PROJECTION_MISSING, "the model produced no projection for this player and event"
    if any((int(key[0]), int(key[1]), int(fid)) in placeholder_pairs for fid in fixtures):
        return OUTCOME_PLACEHOLDER_EXCLUDED, "the stored outcome row is a scheduled placeholder"
    if not any((int(key[0]), int(key[1]), int(fid)) in observed_pairs for fid in fixtures):
        return INPUT_EVIDENCE_UNAVAILABLE, "no stored observation for the target fixture"
    return EVALUATED, ""
