"""PE-3 — legacy-causal structural-bonus evaluation (read-only, deterministic).

ONE canonical path from captured Monte Carlo worlds to PE-3 bonus predictions and metrics.

CLAIM BOUNDARY.  ``evaluation_mode`` is
RETROSPECTIVE_CURRENT_CODE_REPLAY_ON_FROZEN_PREDEADLINE_INPUTS: the PREDICTIVE INPUTS are
genuinely frozen before the target deadline, while the SIMULATION AND CHALLENGER CODE IS
CURRENT.  So it is legitimate to say PE-3 was retrospectively evaluated with current code
against pre-deadline-frozen inputs, and NOT legitimate to say PE-3 predicted the event at
the time, that a historical MC forecast was reproduced, or that the current MC version
equals the historical one.

BACKGROUND AUTHORITY — two distinct modes, never conflated:

  FORMAL_ACCEPTED_GENERATION     a ``bootstrap_generations`` row with accepted = 1 plus its
                                 declared element and persisted counts.
  LEGACY_RECONSTRUCTED_COMPLETE  a fetch that PREDATES the generation-acceptance mechanism,
                                 proven complete by raw-payload <-> snapshot population
                                 identity (see :func:`verify_legacy_bootstrap_generation`).

``LEGACY_RECONSTRUCTED_COMPLETE`` is a RETROSPECTIVE COMPLETENESS CLASSIFICATION, not
``generation_accepted = true``: the acceptance code never ran for such a fetch.  Artefacts
report ``formal_generation_record: NONE`` beside ``legacy_reconstruction`` and emit
``generation_accepted: null`` — never ``true``.  No ``bootstrap_generations`` row is created
or backfilled here.

WHY AGGREGATE HISTORY SUFFICES.  Point-in-time FIXTURE-LEVEL ``player_gameweeks`` history is
unavailable for the GW4 era: the table is upserted in place, so every GW1-GW3 row carries
the most recent fetch's ``updated_at`` and none predates the GW4 cutoff.  That does not
prevent the background: PE-3 needs only cumulative minutes and BPS, which
``player_snapshots`` stores append-only per official fetch, so a player's cumulative rate is
``bps * 90 / minutes`` with no fixture decomposition.  The proof of completeness is the
raw-bootstrap-to-snapshot population identity, which is reproducible and hash-pinned.

DURABLE point-in-time outcome/history retention, and generation-certified provenance for
every predictive freeze, remain deferred to the outcome-capture phase (PE-5).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import bonus_allocation as ba
from . import bonus_challenger as bc
from . import bps_rules as bps
from . import monte_carlo as mc
from . import scoring_rules
from . import walk_forward_metrics as wfm

PE3_EVALUATION_SCHEMA = "pe3_bonus_evaluation_v1.0.0"

EVALUATION_MODE = "RETROSPECTIVE_CURRENT_CODE_REPLAY_ON_FROZEN_PREDEADLINE_INPUTS"

#: Identifies HOW the background evidence was obtained.  Not a model version.
BPS_BACKGROUND_SOURCE_VERSION = "official_bootstrap_cumulative_v1.0.0"

FORMAL_ACCEPTED_GENERATION = "FORMAL_ACCEPTED_GENERATION"
LEGACY_RECONSTRUCTED_COMPLETE = "LEGACY_RECONSTRUCTED_COMPLETE"

#: The accepted shrinkage pseudo-count.  NOT tunable.
SHRINKAGE_MINUTES = 900

#: The declared CURRENT evaluation config.  Not tuned against any result.
CURRENT_REPLAY_SIMULATIONS = 10_000
CURRENT_REPLAY_OCCUPANCY_AUDIT = True

#: Retrospective acceptance thresholds, applied diagnostically only.
LEGACY_RETAINED_FRACTION_MIN = 0.97
LEGACY_ABSOLUTE_DROP_MAX = 75

#: The frozen GW4 raw bootstrap digest, so a later mutation becomes visible.
GW4_LEGACY_RAW_SHA256 = "b2c2cfade784e8f67ae0dcc21db1d3dbed408df110eef662797705b67db21f3c"


class BackgroundEvidenceError(RuntimeError):
    """A historical BPS row that qualifies causally carries no BPS."""


class CaptureContractError(RuntimeError):
    """A captured world does not contain every player the fixture requires."""


class BackgroundAuthorityError(RuntimeError):
    """The background authority could not be established; fail closed."""


# ---------------------------------------------------------------------------
# Legacy bootstrap reconstruction — reproducible, never a hard-coded verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyBootstrapEvidence:
    fetch_run_id: int
    status: str
    raw_path: str
    raw_sha256: str
    raw_elements: int
    snapshots: int
    raw_id_digest: str
    snapshot_id_digest: str
    raw_json_compared: int
    raw_json_matches: int
    captured_at: str
    fetch_finished_at: str
    previous_fetch_run_id: int | None
    previous_population: int | None
    retained_fraction: float | None
    absolute_drop: int | None
    whole_club_loss: bool | None
    authority_mode: str = LEGACY_RECONSTRUCTED_COMPLETE
    formal_generation_record: str = "NONE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetch_run_id": self.fetch_run_id, "status": self.status,
            "raw_path": self.raw_path, "raw_sha256": self.raw_sha256,
            "raw_elements": self.raw_elements, "snapshots": self.snapshots,
            "raw_id_digest": self.raw_id_digest, "snapshot_id_digest": self.snapshot_id_digest,
            "raw_json_compared": self.raw_json_compared,
            "raw_json_matches": self.raw_json_matches,
            "captured_at": self.captured_at, "fetch_finished_at": self.fetch_finished_at,
            "previous_fetch_run_id": self.previous_fetch_run_id,
            "previous_population": self.previous_population,
            "retained_fraction": self.retained_fraction, "absolute_drop": self.absolute_drop,
            "whole_club_loss": self.whole_club_loss,
            "authority_mode": self.authority_mode,
            "formal_generation_record": self.formal_generation_record,
            "generation_accepted": None,
        }


def _id_digest(ids: Iterable[int]) -> str:
    payload = ",".join(str(int(i)) for i in sorted({int(i) for i in ids}))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_legacy_bootstrap_generation(conn: sqlite3.Connection, *, fetch_run_id: int,
                                       raw_dir: str | Path, deadline: str, target_event: int,
                                       expected_raw_sha256: str | None = None,
                                       ) -> LegacyBootstrapEvidence:
    """Prove a pre-generation-era fetch carried a COMPLETE bootstrap population.

    Gates, all of which must pass (fail closed otherwise):

      A. the fetch run exists
      B. status == success
      C. the run-scoped raw bootstrap exists
      D. canonical parse + validation succeed
      E. raw element count == snapshot count for that fetch run
      F. raw element ID set == snapshot player ID set
      G. every snapshot ``raw_json`` semantically equals its raw element
      H. exactly one coherent snapshot ``captured_at``
      I. the fetch finished before the target deadline
      J. every target fixture kicked off after the fetch finished
      K. the population diagnostic passes, or is explicitly recorded
    """

    run = conn.execute("SELECT * FROM fetch_runs WHERE id = ?", (int(fetch_run_id),)).fetchone()
    if run is None:
        raise BackgroundAuthorityError(f"fetch run {fetch_run_id} does not exist")
    status = str(run["status"])
    if status != "success":
        raise BackgroundAuthorityError(f"fetch run {fetch_run_id} status is {status!r}, not success")
    raw_path = Path(raw_dir) / str(int(fetch_run_id)) / "bootstrap_static.json"
    if not raw_path.exists():
        raise BackgroundAuthorityError(f"raw bootstrap missing at {raw_path}")
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_raw_sha256 is not None and raw_sha256 != expected_raw_sha256:
        raise BackgroundAuthorityError(
            f"raw bootstrap sha256 {raw_sha256} does not match the frozen value "
            f"{expected_raw_sha256}; the file changed since it was certified"
        )

    from . import parsers

    payload = json.loads(raw_bytes.decode("utf-8"))
    try:
        parsers.validate_bootstrap_payload(payload)
    except Exception as failure:  # noqa: BLE001 - any rejection is a hard stop
        raise BackgroundAuthorityError(f"canonical bootstrap validation failed: {failure}") from failure
    elements = {int(item["id"]): item for item in (payload.get("elements") or [])}

    rows = list(conn.execute(
        "SELECT player_id, captured_at, raw_json FROM player_snapshots WHERE fetch_run_id = ?",
        (int(fetch_run_id),)))
    if not rows:
        raise BackgroundAuthorityError(f"no player_snapshots for fetch run {fetch_run_id}")
    snapshot_ids = {int(r["player_id"]) for r in rows}
    if len(elements) != len(rows):
        raise BackgroundAuthorityError(
            f"raw element count {len(elements)} != snapshot count {len(rows)}")
    if set(elements) != snapshot_ids:
        missing = sorted(set(elements) - snapshot_ids)[:10]
        extra = sorted(snapshot_ids - set(elements))[:10]
        raise BackgroundAuthorityError(
            f"raw ID set != snapshot ID set (missing {missing}, extra {extra})")
    compared = matches = 0
    for row in rows:
        player_id = int(row["player_id"])
        compared += 1
        if json.loads(row["raw_json"]) == elements[player_id]:
            matches += 1
    if matches != compared:
        raise BackgroundAuthorityError(
            f"{compared - matches} of {compared} snapshots differ from their raw bootstrap element")
    captured = sorted({str(r["captured_at"]) for r in rows})
    if len(captured) != 1:
        raise BackgroundAuthorityError(
            f"snapshot capture is not one coherent instant: {captured[:5]}")
    finished_at = str(run["finished_at"])
    if not (finished_at < str(deadline)):
        raise BackgroundAuthorityError(
            f"fetch finished {finished_at} is not before the deadline {deadline}")
    kickoffs = [str(r[0]) for r in conn.execute(
        "SELECT kickoff_time FROM fixtures WHERE event = ? AND kickoff_time IS NOT NULL",
        (int(target_event),))]
    late = [k for k in kickoffs if k <= finished_at]
    if late:
        raise BackgroundAuthorityError(
            f"{len(late)} target fixture(s) kicked off before the fetch completed; earliest {min(late)}")

    previous_id, previous_ids = _previous_legacy_population(
        conn, fetch_run_id=int(fetch_run_id), raw_dir=raw_dir)
    retained = absolute = None
    whole_club_loss = None
    if previous_ids:
        retained = len(previous_ids & set(elements)) / len(previous_ids)
        absolute = len(previous_ids - set(elements))
        whole_club_loss = _whole_club_loss(conn, removed=previous_ids - set(elements),
                                           surviving=set(elements))
    return LegacyBootstrapEvidence(
        fetch_run_id=int(fetch_run_id), status=status, raw_path=str(raw_path),
        raw_sha256=raw_sha256, raw_elements=len(elements), snapshots=len(rows),
        raw_id_digest=_id_digest(elements), snapshot_id_digest=_id_digest(snapshot_ids),
        raw_json_compared=compared, raw_json_matches=matches, captured_at=captured[0],
        fetch_finished_at=finished_at, previous_fetch_run_id=previous_id,
        previous_population=len(previous_ids) if previous_ids else None,
        retained_fraction=retained, absolute_drop=absolute, whole_club_loss=whole_club_loss)


def _previous_legacy_population(conn: sqlite3.Connection, *, fetch_run_id: int,
                                raw_dir: str | Path) -> tuple[int | None, set[int]]:
    """Nearest EARLIER successful fetch proven to carry a complete raw<->snapshot population."""

    for run in conn.execute(
            "SELECT id FROM fetch_runs WHERE id < ? AND status = 'success' ORDER BY id DESC",
            (int(fetch_run_id),)):
        path = Path(raw_dir) / str(int(run["id"])) / "bootstrap_static.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - an unreadable legacy file is simply skipped
            continue
        element_ids = {int(item["id"]) for item in (payload.get("elements") or [])}
        if not element_ids:
            continue
        snapshot_ids = {int(r[0]) for r in conn.execute(
            "SELECT player_id FROM player_snapshots WHERE fetch_run_id = ?", (int(run["id"]),))}
        if element_ids == snapshot_ids:
            return int(run["id"]), element_ids
    return None, set()


def _whole_club_loss(conn: sqlite3.Connection, *, removed: set[int],
                     surviving: set[int]) -> bool | None:
    """True when every player of some club disappeared between two populations."""

    clubs: dict[int, set[int]] = {}
    for row in conn.execute("SELECT id, team_id FROM players"):
        clubs.setdefault(int(row["team_id"]), set()).add(int(row["id"]))
    for members in clubs.values():
        if members & removed and not (members & surviving):
            return True
    return False


# ---------------------------------------------------------------------------
# Causal background from ONE bootstrap snapshot generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackgroundEvidence:
    player_id: int
    position: str | None
    personal_minutes: int
    personal_bps: int
    raw_personal_per90: float | None
    fallback_scope: str
    fallback_population_players: int
    fallback_population_minutes: int
    fallback_per90: float
    shrinkage_minutes: int
    shrunk_per90: float

    def expected_bps(self, mean_predicted_minutes: float) -> float:
        """Historical level for those PREDICTED minutes (never realised minutes)."""

        return self.shrunk_per90 * float(mean_predicted_minutes) / 90.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _historical_position(snapshot_raw: Mapping[str, Any]) -> str | None:
    """Position from THAT snapshot's own raw_json, never from the current players table."""

    element_type = snapshot_raw.get("element_type")
    if element_type is None:
        return None
    return scoring_rules.POSITION_IDS.get(int(element_type))


def load_snapshot_backgrounds(conn: sqlite3.Connection, *, fetch_run_id: int,
                              shrinkage_minutes: int = SHRINKAGE_MINUTES,
                              ) -> tuple[dict[int, BackgroundEvidence], dict[str, Any]]:
    """Per-player causal BPS level from ONE bootstrap generation.

    Every value comes from the SAME fetch run, so capture instants can never be mixed.  A
    positive-minute row with NULL BPS fails closed.
    """

    rows = list(conn.execute(
        "SELECT player_id, minutes, bps, raw_json FROM player_snapshots WHERE fetch_run_id = ?",
        (int(fetch_run_id),)))
    if not rows:
        raise BackgroundAuthorityError(f"no snapshots for fetch run {fetch_run_id}")

    personal: dict[int, tuple[int, int]] = {}
    positions: dict[int, str] = {}
    for row in rows:
        player_id = int(row["player_id"])
        snapshot = json.loads(row["raw_json"])
        position = _historical_position(snapshot)
        if position is None:
            raise BackgroundAuthorityError(
                f"snapshot for player {player_id} carries no usable historical element_type")
        positions[player_id] = position
        minutes = int(row["minutes"] or 0)
        if minutes <= 0:
            continue
        value = row["bps"]
        if value is None:
            raise BackgroundEvidenceError(
                f"player {player_id} has {minutes} minutes but no BPS; 'unknown' is not zero")
        personal[player_id] = (minutes, int(value))

    league_minutes = sum(m for m, _ in personal.values())
    league_bps = sum(b for _, b in personal.values())
    by_position: dict[str, list[tuple[int, int]]] = {}
    for player_id, entry in personal.items():
        by_position.setdefault(positions[player_id], []).append(entry)
    position_totals = {p: (sum(m for m, _ in v), sum(b for _, b in v))
                       for p, v in by_position.items()}

    evidence: dict[int, BackgroundEvidence] = {}
    for player_id, position in positions.items():
        minutes, bps_value = personal.get(player_id, (0, 0))
        pool_minutes, pool_bps = position_totals.get(position, (0, 0))
        if pool_minutes > 0:
            scope, fb_minutes, fb_bps = "POSITION", pool_minutes, pool_bps
            population = len(by_position[position])
        elif league_minutes > 0:
            scope, fb_minutes, fb_bps = "LEAGUE", league_minutes, league_bps
            population = len(personal)
        else:
            raise BackgroundAuthorityError(
                "no causal league evidence: no positive-minute player carries BPS")
        fallback_per90 = fb_bps * 90.0 / fb_minutes
        if minutes > 0:
            raw_per90 = bps_value * 90.0 / minutes
            weight = minutes / (minutes + float(shrinkage_minutes))
            shrunk = weight * raw_per90 + (1.0 - weight) * fallback_per90
        else:
            raw_per90 = None
            shrunk = fallback_per90
        evidence[player_id] = BackgroundEvidence(
            player_id=player_id, position=position, personal_minutes=minutes,
            personal_bps=bps_value, raw_personal_per90=raw_per90, fallback_scope=scope,
            fallback_population_players=population, fallback_population_minutes=fb_minutes,
            fallback_per90=fallback_per90, shrinkage_minutes=int(shrinkage_minutes),
            shrunk_per90=shrunk)
    provenance = {
        "source_version": BPS_BACKGROUND_SOURCE_VERSION, "fetch_run_id": int(fetch_run_id),
        "snapshot_rows": len(rows), "players": len(evidence),
        "players_with_personal_history": len(personal), "league_pool_minutes": league_minutes,
        "position_pool_minutes": {p: t[0] for p, t in sorted(position_totals.items())},
        "shrinkage_minutes": int(shrinkage_minutes),
    }
    return evidence, provenance


# ---------------------------------------------------------------------------
# The one end-to-end orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PlayerFixturePrediction:
    player_id: int
    fixture_id: int
    position: str | None
    expected_minutes: float
    background_bps_per90: float
    background_expected_bps: float
    expected_bonus: float
    p_bonus_any: float
    p_bonus_2plus: float
    p_bonus_3: float
    mean_bps_proxy: float
    unsupported_rule_rows: tuple[str, ...] = ()
    limitation_flags: tuple[str, ...] = ()
    background: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id, "fixture_id": self.fixture_id,
            "position": self.position, "expected_minutes": self.expected_minutes,
            "background_bps_per90": self.background_bps_per90,
            "background_expected_bps": self.background_expected_bps,
            "expected_bonus": self.expected_bonus, "p_bonus_any": self.p_bonus_any,
            "p_bonus_2plus": self.p_bonus_2plus, "p_bonus_3": self.p_bonus_3,
            "mean_bps_proxy": self.mean_bps_proxy,
            "unsupported_rule_rows": list(self.unsupported_rule_rows),
            "limitation_flags": list(self.limitation_flags), "background": self.background,
        }


def evaluate_fixture_bonus_worlds(*, fixture_id: int,
                                  captured_worlds: Sequence[Mapping[int, Mapping[str, Any]]],
                                  backgrounds: Mapping[int, BackgroundEvidence],
                                  rules: Sequence[bps.BPSPrimitiveSpec] = bps.RULE_SPECS,
                                  ) -> tuple[list[PlayerFixturePrediction], dict[str, Any]]:
    """THE canonical PE-3 path for one fixture: captured worlds -> bonus predictions.

    The WHOLE fixture competes in every world.  Proxy ordering is exact numeric: no
    rounding, no quantisation, no epsilon.
    """

    if not captured_worlds:
        return [], {"fixtures": 0, "worlds": 0, "players": 0}

    players = sorted({int(pid) for world in captured_worlds for pid in world})
    structural: dict[int, list[float]] = {pid: [] for pid in players}
    minutes = {pid: 0.0 for pid in players}
    positions: dict[int, str | None] = {pid: None for pid in players}
    unsupported_by_player: dict[int, set[str]] = {pid: set() for pid in players}
    flags_by_player: dict[int, set[str]] = {pid: set() for pid in players}

    for index, world in enumerate(captured_worlds):
        for player_id in players:
            record = world.get(player_id)
            if record is None:
                raise CaptureContractError(
                    f"captured world {index} for fixture {fixture_id} has no record for "
                    f"player {player_id}; the capture contract requires every simulated "
                    "player in every world, so this is malformed capture, not a zero")
            positions[player_id] = record.get("position") or positions[player_id]
            minutes[player_id] += float(record["minutes"])
            proxy = bc.structural_world_bps(player_id, record["position"] or "MID", record, rules)
            structural[player_id].append(float(proxy.bps))
            unsupported_by_player[player_id].update(proxy.unsupported_rule_rows)
            flags_by_player[player_id].update(proxy.flags)

    worlds = len(captured_worlds)
    mean_minutes = {pid: minutes[pid] / worlds for pid in players}
    proxy_by_world: list[dict[int, float]] = [{} for _ in range(worlds)]
    background_expected: dict[int, float] = {}
    background_per90: dict[int, float] = {}
    for player_id in players:
        evidence = backgrounds.get(player_id)
        if evidence is None:
            raise BackgroundEvidenceError(f"no causal background for player {player_id}")
        background_per90[player_id] = evidence.shrunk_per90
        expected = evidence.expected_bps(mean_minutes[player_id])
        background_expected[player_id] = expected
        for position_index, value in enumerate(
                bc.centred_world_bps(expected, structural[player_id])):
            proxy_by_world[position_index][player_id] = value

    summary = bc.aggregate_worlds(proxy_by_world)
    predictions: list[PlayerFixturePrediction] = []
    for player_id in players:
        predictions.append(PlayerFixturePrediction(
            player_id=player_id, fixture_id=int(fixture_id), position=positions[player_id],
            expected_minutes=mean_minutes[player_id],
            background_bps_per90=background_per90[player_id],
            background_expected_bps=background_expected[player_id],
            expected_bonus=summary.expected_bonus[player_id],
            p_bonus_any=summary.p_bonus_any[player_id],
            p_bonus_2plus=summary.p_bonus_2plus[player_id],
            p_bonus_3=summary.p_bonus_3[player_id],
            mean_bps_proxy=summary.mean_bps_proxy[player_id],
            unsupported_rule_rows=tuple(sorted(unsupported_by_player[player_id])),
            limitation_flags=tuple(sorted(flags_by_player[player_id])),
            background=backgrounds[player_id].as_dict()))
    diagnostics = _world_diagnostics(
        fixture_id=int(fixture_id), proxy_by_world=proxy_by_world, players=players,
        summary=summary)
    diagnostics["unsupported_rule_rows_union"] = sorted(
        {row for rows in unsupported_by_player.values() for row in rows})
    diagnostics["limitation_flags_union"] = sorted(
        {flag for rows in flags_by_player.values() for flag in rows})
    return predictions, diagnostics


def _world_diagnostics(*, fixture_id: int, proxy_by_world: Sequence[Mapping[int, float]],
                       players: Sequence[int], summary: Any) -> dict[str, Any]:
    """Per-world tie structure plus one PROVEN rival-dependence example."""

    any_tie = bonus_tie = over_six = 0
    rival_example: dict[str, Any] | None = None
    for world in proxy_by_world:
        counts: dict[float, int] = {}
        for value in world.values():
            counts[value] = counts.get(value, 0) + 1
        if any(count > 1 for count in counts.values()):
            any_tie += 1
        allocation = ba.allocate_fixture_bonus(world)
        if sum(allocation.values()) > 6:
            over_six += 1
        if any(bonus > 0 and counts[world[pid]] > 1 for pid, bonus in allocation.items()):
            bonus_tie += 1
        if rival_example is None and len(players) >= 2:
            me = players[0]
            for other in players[1:]:
                # MY proxy is untouched; only a rival's changes.
                shifted = dict(world)
                shifted[other] = shifted[other] + 1000.0
                after = ba.allocate_fixture_bonus(shifted)[me]
                if after != allocation[me]:
                    rival_example = {
                        "fixture_id": fixture_id, "player_id": int(me), "rival_id": int(other),
                        "player_proxy_unchanged": world[me],
                        "rival_proxy_before": world[other],
                        "rival_proxy_after": shifted[other],
                        "player_bonus_before": allocation[me], "player_bonus_after": after}
                    break
    worlds = len(proxy_by_world)
    return {
        "fixtures": 1, "worlds": worlds, "players": len(players),
        "any_proxy_tie_worlds": any_tie,
        "any_proxy_tie_rate": any_tie / worlds if worlds else 0.0,
        "bonus_affecting_tie_worlds": bonus_tie,
        "bonus_affecting_tie_rate": bonus_tie / worlds if worlds else 0.0,
        "worlds_over_six_bonus": over_six,
        "ranking_changed_across_worlds": bool(summary.ranking_changed_across_worlds),
        "rival_dependence_example": rival_example,
    }


# ---------------------------------------------------------------------------
# Outcomes and metrics
# ---------------------------------------------------------------------------


def _is_placeholder(record: Mapping[str, Any]) -> bool:
    """The repository's scheduled-placeholder signature: minutes only, nothing realised."""

    if record.get("bonus") is not None or record.get("bps") is not None:
        return False
    return record.get("updated_at") is None or record.get("starts") is None


def final_outcomes(conn: sqlite3.Connection, *, event: int) -> tuple[dict[tuple[int, int], dict], dict]:
    """Final official outcomes for one FINAL, CHECKED event.

    A scheduled placeholder is not a realised zero: it is EXCLUDED AND COUNTED, as is a
    genuine missing bonus.
    """

    rows = conn.execute(
        """
        SELECT pg.player_id AS player_id, pg.fixture_id AS fixture_id, pg.bonus AS bonus,
               pg.minutes AS minutes, pg.bps AS bps, pg.starts AS starts,
               pg.updated_at AS updated_at
        FROM player_gameweeks pg
        JOIN fixtures f ON f.id = pg.fixture_id
        JOIN events e ON e.id = f.event
        WHERE f.event = ? AND f.finished = 1 AND e.finished = 1 AND e.data_checked = 1
        """,
        (int(event),)).fetchall()
    outcomes: dict[tuple[int, int], dict] = {}
    excluded = {"placeholder": 0, "missing_bonus": 0}
    for row in rows:
        record = dict(row)
        if _is_placeholder(record):
            excluded["placeholder"] += 1
            continue
        if record.get("bonus") is None:
            excluded["missing_bonus"] += 1
            continue
        outcomes[(int(row["player_id"]), int(row["fixture_id"]))] = record
    return outcomes, excluded


def _metric(metric: Any) -> dict[str, Any]:
    """Preserve value/status/n/detail rather than collapsing to a naked float."""

    if hasattr(metric, "as_dict"):
        return metric.as_dict()
    return {"value": getattr(metric, "value", metric), "status": "OK", "n": None, "detail": None}


def score_predictions(predictions: Sequence[PlayerFixturePrediction],
                      outcomes: Mapping[tuple[int, int], Mapping[str, Any]],
                      ) -> dict[str, Any]:
    """PE-2 metric functions over the eligible population."""

    eligible: list[tuple[PlayerFixturePrediction, int]] = []
    excluded = {"missing_outcome": 0}
    for prediction in predictions:
        outcome = outcomes.get((prediction.player_id, prediction.fixture_id))
        if outcome is None:
            excluded["missing_outcome"] += 1
            continue
        eligible.append((prediction, int(outcome["bonus"])))
    if not eligible:
        return {"status": "NO_SAMPLE", "n": 0, "excluded": excluded}

    keys = [(p.player_id, p.fixture_id) for p, _ in eligible]
    predicted = [p.expected_bonus for p, _ in eligible]
    realised = [float(b) for _, b in eligible]
    report: dict[str, Any] = {
        "status": "OK", "n": len(eligible), "population_digest": population_digest(keys),
        "mean_predicted_bonus": sum(predicted) / len(predicted),
        "mean_realised_bonus": sum(realised) / len(realised),
        "bias": _metric(wfm.mean_bias(predicted, realised)),
        "mae": _metric(wfm.mean_absolute_error(predicted, realised)),
        "brier_p_any": _metric(wfm.brier_score(
            [p.p_bonus_any for p, _ in eligible], [1.0 if b > 0 else 0.0 for _, b in eligible])),
        "excluded": excluded, "by_position": {},
    }
    for position in ("GKP", "DEF", "MID", "FWD"):
        group = [(p, b) for p, b in eligible if p.position == position]
        if not group:
            report["by_position"][position] = {"status": "NO_SAMPLE", "n": 0}
            continue
        gp = [p.expected_bonus for p, _ in group]
        gr = [float(b) for _, b in group]
        report["by_position"][position] = {
            "status": "OK", "n": len(group),
            "mean_predicted": sum(gp) / len(gp), "mean_realised": sum(gr) / len(gr),
            "bias": _metric(wfm.mean_bias(gp, gr)),
            "mae": _metric(wfm.mean_absolute_error(gp, gr)),
            "brier_p_any": _metric(wfm.brier_score(
                [p.p_bonus_any for p, _ in group], [1.0 if b > 0 else 0.0 for _, b in group])),
        }
    return report


def soft_baseline(conn: sqlite3.Connection, *, xpts_run_id: int, keys: Iterable[tuple[int, int]],
                  outcomes: Mapping[tuple[int, int], Mapping[str, Any]]) -> dict[str, Any]:
    """The certified soft-bonus path on EXACTLY the same eligible key set."""

    key_list = sorted({(int(p), int(f)) for p, f in keys})
    if not key_list:
        return {"status": "NO_SAMPLE", "n": 0}
    values: dict[tuple[int, int], float] = {}
    for player_id, fixture_id in key_list:
        row = conn.execute(
            "SELECT payload_json FROM player_fixture_xpts_projections "
            "WHERE projection_run_id = ? AND player_id = ? AND fixture_id = ?",
            (int(xpts_run_id), player_id, fixture_id)).fetchone()
        if row is None or row["payload_json"] is None:
            return {"status": "INCOMPLETE_BASELINE", "n": 0,
                    "missing": {"player_id": player_id, "fixture_id": fixture_id}}
        payload = json.loads(row["payload_json"])
        if payload.get("bonus_xpts") is None:
            return {"status": "INCOMPLETE_BASELINE", "n": 0,
                    "missing": {"player_id": player_id, "fixture_id": fixture_id}}
        values[(player_id, fixture_id)] = float(payload["bonus_xpts"])
    predicted = [values[k] for k in key_list]
    realised = [float(outcomes[k]["bonus"]) for k in key_list]
    return {
        "status": "OK", "n": len(key_list), "population_digest": population_digest(key_list),
        "mean_predicted": sum(predicted) / len(predicted),
        "mean_realised": sum(realised) / len(realised),
        "bias": _metric(wfm.mean_bias(predicted, realised)),
        "mae": _metric(wfm.mean_absolute_error(predicted, realised)),
        "brier": "NOT_APPLICABLE — expected points is not a probability",
    }


def population_digest(keys: Iterable[tuple[int, int]]) -> str:
    payload = ",".join(f"{int(p)}:{int(f)}" for p, f in sorted({(int(a), int(b)) for a, b in keys}))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_vs_gameweek_diagnostic(conn: sqlite3.Connection, *, fetch_run_id: int,
                                    through_event: int = 3) -> dict[str, Any]:
    """RETROSPECTIVE integrity diagnostic ONLY — never model input."""

    snaps = {int(r["player_id"]): (int(r["minutes"] or 0), r["bps"])
             for r in conn.execute(
                 "SELECT player_id, minutes, bps FROM player_snapshots WHERE fetch_run_id = ?",
                 (int(fetch_run_id),))}
    sums: dict[int, list[int]] = {}
    for row in conn.execute(
            """SELECT pg.player_id AS player_id, SUM(pg.minutes) AS minutes, SUM(pg.bps) AS bps
               FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
               WHERE f.event <= ? AND f.finished = 1 GROUP BY pg.player_id""",
            (int(through_event),)):
        sums[int(row["player_id"])] = [int(row["minutes"] or 0), int(row["bps"] or 0)]
    comparable = minute_matches = bps_matches = 0
    examples: list[dict[str, Any]] = []
    for player_id, (snapshot_minutes, snapshot_bps) in snaps.items():
        if player_id not in sums or snapshot_bps is None:
            continue
        comparable += 1
        gw_minutes, gw_bps = sums[player_id]
        if snapshot_minutes == gw_minutes:
            minute_matches += 1
        if snapshot_bps == gw_bps:
            bps_matches += 1
        elif len(examples) < 5:
            examples.append({"player_id": player_id, "snapshot_minutes": snapshot_minutes,
                             "gw_minutes": gw_minutes, "snapshot_bps": snapshot_bps,
                             "gw_bps": gw_bps})
    return {"label": "RETROSPECTIVE_INTEGRITY_DIAGNOSTIC_ONLY", "comparable_players": comparable,
            "exact_minute_matches": minute_matches, "exact_bps_matches": bps_matches,
            "minute_mismatches": comparable - minute_matches,
            "bps_mismatches": comparable - bps_matches, "examples": examples}


def _calibration_state_identity(config: "mc.MonteCarloConfig") -> str:
    if not hasattr(mc, "calibration_provenance"):
        raise CaptureContractError("monte_carlo.calibration_provenance is unavailable")
    provenance = mc.calibration_provenance(config)
    identity = provenance.get("calibration_state_identity")
    if not identity:
        raise CaptureContractError(
            "calibration provenance carries no calibration_state_identity; refusing to "
            f"emit a silent None (keys: {sorted(provenance)})")
    return str(identity)


def evaluate_event(conn: sqlite3.Connection, *, event: int, as_of: str, deadline: str,
                   chain: Mapping[str, int], fetch_run_id: int, raw_dir: str | Path,
                   target_event: int, xpts_run_id: int,
                   mc_config: "mc.MonteCarloConfig | None" = None,
                   rules: Sequence[bps.BPSPrimitiveSpec] = bps.RULE_SPECS,
                   shrinkage_minutes: int = SHRINKAGE_MINUTES,
                   expected_raw_sha256: str | None = None,
                   preloaded_worlds: Mapping[int, Any] | None = None) -> dict[str, Any]:
    """THE full PE-3 evaluation path.  READ ONLY — nothing is persisted."""

    problems = mc.validate_input_run_coherence(
        conn, xpts_run_id=int(chain["xpts"]), minutes_run_id=int(chain["minutes"]),
        team_run_id=int(chain["team"]), rate_run_id=int(chain["rate"]))
    if problems:
        raise BackgroundAuthorityError(f"input chain is incoherent: {problems}")

    evidence = verify_legacy_bootstrap_generation(
        conn, fetch_run_id=int(fetch_run_id), raw_dir=raw_dir, deadline=deadline,
        target_event=int(target_event), expected_raw_sha256=expected_raw_sha256)

    config = mc_config or mc.MonteCarloConfig(
        simulations=CURRENT_REPLAY_SIMULATIONS, occupancy_audit=CURRENT_REPLAY_OCCUPANCY_AUDIT)
    if preloaded_worlds is None:
        fixtures = mc.load_fixture_inputs(
            conn, event=int(event), xpts_run_id=int(chain["xpts"]),
            minutes_run_id=int(chain["minutes"]), team_run_id=int(chain["team"]))
        result = mc.simulate(fixtures, config, scoring_rules.DEFAULT_SCORING_RULES,
                             capture_bps_worlds=True)
        captured = result["bps_worlds"]
    else:
        captured = preloaded_worlds

    backgrounds, background_provenance = load_snapshot_backgrounds(
        conn, fetch_run_id=int(fetch_run_id), shrinkage_minutes=int(shrinkage_minutes))

    predictions: list[PlayerFixturePrediction] = []
    per_fixture: dict[str, Any] = {}
    for fixture_id in sorted(captured):
        rows, diagnostics = evaluate_fixture_bonus_worlds(
            fixture_id=int(fixture_id), captured_worlds=captured[fixture_id],
            backgrounds=backgrounds, rules=rules)
        predictions.extend(rows)
        per_fixture[str(fixture_id)] = diagnostics

    outcomes, outcome_exclusions = final_outcomes(conn, event=int(event))
    structural = score_predictions(predictions, outcomes)
    if structural.get("status") == "OK":
        eligible_keys = [(p.player_id, p.fixture_id) for p in predictions
                         if (p.player_id, p.fixture_id) in outcomes]
        soft = soft_baseline(conn, xpts_run_id=int(xpts_run_id), keys=eligible_keys,
                             outcomes=outcomes)
    else:
        soft = {"status": "NO_SAMPLE", "n": 0}
    same_population = bool(
        structural.get("status") == "OK" and soft.get("status") == "OK"
        and structural["population_digest"] == soft["population_digest"])

    world_diagnostics = {
        "fixtures": len(per_fixture),
        "worlds_per_fixture": {k: v["worlds"] for k, v in sorted(per_fixture.items())},
        "player_fixture_predictions": len(predictions),
        "any_proxy_tie_worlds": sum(v.get("any_proxy_tie_worlds", 0) for v in per_fixture.values()),
        "bonus_affecting_tie_worlds": sum(
            v.get("bonus_affecting_tie_worlds", 0) for v in per_fixture.values()),
        "worlds_over_six_bonus": sum(v.get("worlds_over_six_bonus", 0) for v in per_fixture.values()),
        "rival_dependence_examples": [v["rival_dependence_example"] for v in per_fixture.values()
                                      if v.get("rival_dependence_example")][:3],
        "highest_expected_bonus": sorted(
            ({"player_id": p.player_id, "fixture_id": p.fixture_id, "expected_bonus": p.expected_bonus}
             for p in predictions), key=lambda item: -item["expected_bonus"])[:3],
        "lowest_non_zero_expected_bonus": sorted(
            ({"player_id": p.player_id, "fixture_id": p.fixture_id, "expected_bonus": p.expected_bonus}
             for p in predictions if p.expected_bonus > 0),
            key=lambda item: item["expected_bonus"])[:3],
        "limitation_flags_union": sorted(
            {flag for v in per_fixture.values() for flag in v.get("limitation_flags_union", [])}),
        "unsupported_rule_rows_union": sorted(
            {row for v in per_fixture.values() for row in v.get("unsupported_rule_rows_union", [])}),
    }

    return {
        "schema": PE3_EVALUATION_SCHEMA,
        "evaluation_mode": EVALUATION_MODE,
        "target_event": int(event),
        "historical_input_identity": {
            "cutoff": as_of, "deadline": deadline,
            "xpts_run_id": int(chain["xpts"]), "minutes_run_id": int(chain["minutes"]),
            "team_run_id": int(chain["team"]), "rate_run_id": int(chain["rate"]),
            "official_fetch_run_id": int(fetch_run_id),
            "background_authority": LEGACY_RECONSTRUCTED_COMPLETE,
            "formal_generation_record": "NONE",
            "legacy_reconstruction": evidence.as_dict(),
        },
        "current_replay_identity": {
            "mc_version": mc.MONTE_CARLO_MODEL_VERSION,
            "config_hash": config.config_hash(), "seed": int(config.seed),
            "simulations": int(config.simulations),
            "calibration_state_identity": _calibration_state_identity(config),
        },
        "bps_rules": bps.ruleset_fingerprint(),
        "background_source_version": BPS_BACKGROUND_SOURCE_VERSION,
        "challenger_version": bc.BONUS_BPS_MODEL_VERSION,
        "background_provenance": background_provenance,
        "structural": structural,
        "soft_baseline": soft,
        "same_population": same_population,
        "outcome_exclusions": outcome_exclusions,
        "world_diagnostics": world_diagnostics,
        "snapshot_integrity_diagnostic": snapshot_vs_gameweek_diagnostic(
            conn, fetch_run_id=int(fetch_run_id)),
        "claim": "DESCRIPTIVE_ONLY",
        "limitations": [
            "STRUCTURAL APPROXIMATION: the proxy models only minutes, goals, assists, clean "
            "sheets, goals conceded, saves and yellow cards; 28 rule rows are unsupported and "
            "named per player.",
            "RETROSPECTIVE: current MC and challenger code against frozen pre-deadline inputs. "
            "Not a historical frozen PE-3 forecast, and not a reproduction of the historical MC run.",
            "NO MODEL-SELECTION VERDICT: this is a tiny sample.",
        ],
    }
