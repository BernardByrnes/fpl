"""PE-3 — causal structural-bonus evaluation (read-only, deterministic).

ONE canonical path from captured Monte Carlo worlds to PE-3 bonus predictions and metrics.
A caller must not have to compose the structural proxy, the centring and the allocator by
hand.

CLAIM BOUNDARY — read this before quoting any number this module produces.

    evaluation_mode = RETROSPECTIVE_CURRENT_CODE_REPLAY_ON_FROZEN_PREDEADLINE_INPUTS

The PREDICTIVE INPUTS are genuinely frozen before the target deadline (Minutes, Team,
Player Rates, xPts, all PRE_DEADLINE and all at one authoritative cutoff).  The SIMULATION
AND CHALLENGER CODE IS CURRENT.  So:

  * allowed: "PE-3 was retrospectively evaluated using current challenger/current MC code
    against predictive inputs that were genuinely frozen before the target-event deadline";
  * NOT allowed: "PE-3 predicted GW4 at the time";
  * NOT allowed: "the historical MC forecast was reproduced";
  * NOT allowed: "current mc_v1.3.0 equals historical mc_v1.2.1".

THE CAUSAL BACKGROUND comes only from ``fpl_brain.historical_observations``, the frozen
PE-1 reader, with an explicit ``as_of`` and ``planning_event``.  No history predicate is
restated here.

A NULL BPS on an otherwise-valid positive-minute historical row is UNKNOWN, not zero: it
raises :class:`BackgroundEvidenceError`.  A genuine ``bps == 0`` is valid evidence.

MEASURED LIMITATION — TWO separate facts, and only the first is now fatal.

1. FIXTURE-LEVEL player_gameweeks history is NOT recoverable point-in-time: the table is
   upserted in place, so at the time of writing every GW1-GW3 row carried
   ``updated_at = 2026-09-18T13:53:05Z`` and ZERO rows predated the GW4 authoritative
   cutoff.  That alone is NOT fatal to PE-3: this module needs only AGGREGATE cumulative
   minutes and BPS, which ``player_snapshots`` stores append-only per official fetch.

2. FATAL HERE: the frozen GW4 chain's official fetch is run **36**
   (2026-09-11T22:45:41Z, status success), but ``bootstrap_generations`` does not begin
   until fetch run **38** (2026-09-13T12:34:57Z).  The generation-acceptance mechanism did
   not exist when the GW4 chain was frozen, so for run 36 there is NO generation row and
   therefore NO ``accepted`` flag and NO ``official_element_count`` / ``persisted_count``
   to certify capture completeness.  656 snapshot rows exist for run 36, and without the
   generation record there is no way to tell "that era's element set was 656" from "three
   snapshots failed to persist" — which is exactly what the acceptance record exists to
   decide.  The next generation (id 1, fetch run 38) post-dates the GW4 deadline and GW4
   itself, so using it would leak target-event outcomes into the cumulative totals.

   The evaluation is therefore BLOCKED, and the fix is a durable outcome/history capture
   in the outcome-capture phase (PE-5), not a weaker binding here.

"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import bonus_allocation as ba
from . import bonus_challenger as bc
from . import bps_rules as bps
from . import historical_observations as ho
from . import monte_carlo as mc
from . import scoring_rules
from . import walk_forward_metrics as wfm

PE3_EVALUATION_SCHEMA = "pe3_bonus_evaluation_v1.0.0"

EVALUATION_MODE = "RETROSPECTIVE_CURRENT_CODE_REPLAY_ON_FROZEN_PREDEADLINE_INPUTS"

#: The accepted shrinkage pseudo-count.  NOT tunable — do not adjust.
SHRINKAGE_MINUTES = 900

#: The declared CURRENT evaluation config.  Not tuned against any result.
CURRENT_REPLAY_SIMULATIONS = 10_000


class BackgroundEvidenceError(RuntimeError):
    """A historical BPS row that qualifies causally carries no BPS."""


class CaptureContractError(RuntimeError):
    """A captured world does not contain every player the fixture requires."""


def position_by_player(conn: sqlite3.Connection,
                       player_ids: Iterable[int] | None = None) -> dict[int, str]:
    """Canonical player -> GKP/DEF/MID/FWD, from ``players.element_type``."""

    sql = "SELECT id, element_type FROM players"
    params: list[Any] = []
    ids = sorted({int(p) for p in (player_ids or ())})
    if ids:
        sql += f" WHERE id IN ({','.join('?' for _ in ids)})"
        params.extend(ids)
    positions: dict[int, str] = {}
    for row in conn.execute(sql, tuple(params)):
        element_type = row["element_type"] if not isinstance(row, tuple) else row[1]
        player_id = row["id"] if not isinstance(row, tuple) else row[0]
        position = scoring_rules.POSITION_IDS.get(int(element_type)) if element_type is not None else None
        if position:
            positions[int(player_id)] = str(position)
    return positions


# ---------------------------------------------------------------------------
# Causal background
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackgroundEvidence:
    """One player's causal BPS level, with the pool it was shrunk toward."""

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
        """The historical level for those PREDICTED minutes (never realised minutes)."""

        return self.shrunk_per90 * float(mean_predicted_minutes) / 90.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id, "position": self.position,
            "personal_minutes": self.personal_minutes, "personal_bps": self.personal_bps,
            "raw_personal_per90": self.raw_personal_per90,
            "fallback_scope": self.fallback_scope,
            "fallback_population_players": self.fallback_population_players,
            "fallback_population_minutes": self.fallback_population_minutes,
            "fallback_per90": self.fallback_per90,
            "shrinkage_minutes": self.shrinkage_minutes,
            "shrunk_per90": self.shrunk_per90,
        }


def _summarise_pool(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    minutes = 0
    bps_total = 0
    for row in rows:
        row_minutes = int(row.get("minutes") or 0)
        if row_minutes <= 0:
            continue
        value = row.get("bps")
        if value is None:
            raise BackgroundEvidenceError(
                "a positive-minute historical observation carries no BPS; "
                "'unknown' must never be read as zero"
            )
        minutes += row_minutes
        bps_total += int(value)
    return minutes, bps_total


def load_causal_backgrounds(conn: sqlite3.Connection, *, as_of: str, planning_event: int,
                            player_ids: Sequence[int] | None = None,
                            shrinkage_minutes: int = SHRINKAGE_MINUTES,
                            ) -> tuple[dict[int, BackgroundEvidence], dict[str, Any]]:
    """Per-player causal BPS level, built ONLY from the frozen PE-1 reader.

    ``planning_event`` makes the reader apply its own strictly-earlier-event clause, so the
    TARGET EVENT CANNOT ENTER ITS OWN BACKGROUND — the exclusion is the reader's, not a
    second predicate written here.
    """

    rows = ho.historical_player_fixtures(conn, as_of=as_of, planning_event=planning_event)
    positions = position_by_player(conn)

    personal: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        personal.setdefault(int(row["player_id"]), []).append(row)

    league_minutes, league_bps = _summarise_pool(rows)
    by_position: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        position = positions.get(int(row["player_id"]))
        if position:
            by_position.setdefault(position, []).append(row)
    position_totals = {position: _summarise_pool(group) for position, group in by_position.items()}

    wanted = sorted({int(p) for p in player_ids}) if player_ids is not None else sorted(personal)
    evidence: dict[int, BackgroundEvidence] = {}
    for player_id in wanted:
        own_rows = personal.get(player_id, [])
        personal_minutes, personal_bps = _summarise_pool(own_rows)
        position = positions.get(player_id)
        pool_minutes, pool_bps = position_totals.get(position, (0, 0)) if position else (0, 0)
        if pool_minutes > 0:
            scope, fallback_minutes, fallback_bps = "POSITION", pool_minutes, pool_bps
            population = len({int(r["player_id"]) for r in by_position.get(position, [])})
        elif league_minutes > 0:
            scope, fallback_minutes, fallback_bps = "LEAGUE", league_minutes, league_bps
            population = len(personal)
        else:
            raise BackgroundEvidenceError(
                f"no causal historical pool exists at as_of={as_of} planning_event={planning_event}"
            )
        fallback_per90 = fallback_bps * 90.0 / fallback_minutes
        if personal_minutes > 0:
            raw_per90 = personal_bps * 90.0 / personal_minutes
            weight = personal_minutes / (personal_minutes + float(shrinkage_minutes))
            shrunk = weight * raw_per90 + (1.0 - weight) * fallback_per90
        else:
            raw_per90 = None
            shrunk = fallback_per90
        evidence[player_id] = BackgroundEvidence(
            player_id=player_id, position=position, personal_minutes=personal_minutes,
            personal_bps=personal_bps, raw_personal_per90=raw_per90, fallback_scope=scope,
            fallback_population_players=population, fallback_population_minutes=fallback_minutes,
            fallback_per90=fallback_per90, shrinkage_minutes=int(shrinkage_minutes),
            shrunk_per90=shrunk,
        )
    provenance = {
        "reader": "historical_observations.historical_player_fixtures",
        "as_of": as_of, "planning_event": int(planning_event),
        "source_rows": len(rows),
        "distinct_players_with_history": len(personal),
        "league_pool_minutes": league_minutes,
        "position_pool_minutes": {position: totals[0] for position, totals in sorted(position_totals.items())},
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
            "limitation_flags": list(self.limitation_flags),
            "background": self.background,
        }


def evaluate_fixture_bonus_worlds(*, fixture_id: int, captured_worlds: Sequence[Mapping[int, Mapping[str, Any]]],
                                  backgrounds: Mapping[int, BackgroundEvidence],
                                  rules: Sequence[bps.BPSPrimitiveSpec] = bps.RULE_SPECS,
                                  ) -> tuple[list[PlayerFixturePrediction], dict[str, Any]]:
    """THE canonical PE-3 path for one fixture: captured worlds -> bonus predictions.

    The WHOLE fixture competes in every world — no filtering to owned, active, minutes>0 or
    outcome-bearing players.  BPS proxy ordering is exact numeric; no rounding, no epsilon.
    """

    if not captured_worlds:
        return [], {"fixtures": 0, "worlds": 0}

    players = sorted({int(pid) for world in captured_worlds for pid in world})
    structural: dict[int, list[float]] = {pid: [] for pid in players}
    minutes: dict[int, float] = {pid: 0.0 for pid in players}
    positions: dict[int, str | None] = {pid: None for pid in players}
    # PER PLAYER: one goalkeeper's unsupported save-location row must not become every
    # outfield player's limitation.
    unsupported_by_player: dict[int, set[str]] = {pid: set() for pid in players}
    flags_by_player: dict[int, set[str]] = {pid: set() for pid in players}

    for index, world in enumerate(captured_worlds):
        for player_id in players:
            record = world.get(player_id)
            if record is None:
                # The Step-1 capture guarantees the WHOLE fixture universe in every world,
                # so an absent record means the capture is malformed.  Defaulting it to 0
                # would silently invent a zero-BPS performance for a real player.
                raise CaptureContractError(
                    f"captured world {index} for fixture {fixture_id} has no record for "
                    f"player {player_id}; the capture contract requires every simulated "
                    "player in every world, so this is malformed capture, not a zero"
                )
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
        series = bc.centred_world_bps(expected, structural[player_id])
        for index, value in enumerate(series):
            proxy_by_world[index][player_id] = value

    summary = bc.aggregate_worlds(proxy_by_world)
    predictions: list[PlayerFixturePrediction] = []
    for player_id in players:
        evidence = backgrounds[player_id]
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
            background=evidence.as_dict(),
        ))
    diagnostics = {
        "fixtures": 1, "worlds": worlds, "players": len(players),
        # fixture-level UNION is diagnostics only; each player carries his own set
        "unsupported_rule_rows_union": sorted({row for rows in unsupported_by_player.values() for row in rows}),
        "limitation_flags_union": sorted({flag for rows in flags_by_player.values() for flag in rows}),
        "ranking_changed_across_worlds": summary.ranking_changed_across_worlds,
        "total_bonus_per_world": list(summary.total_bonus_per_world),
    }
    return predictions, diagnostics


def evaluate_event(conn: sqlite3.Connection, *, event: int, as_of: str, chain: Mapping[str, int],
                   mc_config: mc.MonteCarloConfig | None = None,
                   rules: Sequence[bps.BPSPrimitiveSpec] = bps.RULE_SPECS,
                   shrinkage_minutes: int = SHRINKAGE_MINUTES,
                   preloaded_worlds: Mapping[int, Any] | None = None,
                   ) -> dict[str, Any]:
    """Full PE-3 evaluation for one final event from one frozen pre-deadline input chain.

    READ ONLY: the simulation result is never persisted.
    """

    config = mc_config or mc.MonteCarloConfig(simulations=CURRENT_REPLAY_SIMULATIONS,
                                              occupancy_audit=True)
    if preloaded_worlds is None:
        fixtures = mc.load_fixture_inputs(conn, event=int(event), xpts_run_id=int(chain["xpts"]),
                                          minutes_run_id=int(chain["minutes"]),
                                          team_run_id=int(chain["team"]))
        result = mc.simulate(fixtures, config, scoring_rules.DEFAULT_SCORING_RULES,
                             capture_bps_worlds=True)
        captured = result["bps_worlds"]
    else:
        captured = preloaded_worlds

    player_ids = sorted({int(pid) for worlds in captured.values() for world in worlds for pid in world})
    backgrounds, background_provenance = load_causal_backgrounds(
        conn, as_of=as_of, planning_event=int(event), player_ids=player_ids,
        shrinkage_minutes=shrinkage_minutes)

    predictions: list[PlayerFixturePrediction] = []
    diagnostics: dict[str, Any] = {"fixtures": 0, "worlds": 0, "per_fixture": {}}
    for fixture_id in sorted(captured):
        rows, per_fixture = evaluate_fixture_bonus_worlds(
            fixture_id=int(fixture_id), captured_worlds=captured[fixture_id],
            backgrounds=backgrounds, rules=rules)
        predictions.extend(rows)
        diagnostics["fixtures"] += 1
        diagnostics["worlds"] = max(diagnostics["worlds"], per_fixture["worlds"])
        diagnostics["per_fixture"][str(fixture_id)] = per_fixture
    diagnostics["max_source_event"] = _max_background_event(conn, as_of=as_of, planning_event=int(event))
    return {
        "predictions": predictions,
        "diagnostics": diagnostics,
        "background_provenance": background_provenance,
        "current_replay_identity": {
            "mc_version": mc.MONTE_CARLO_MODEL_VERSION,
            "config_hash": config.config_hash(),
            "seed": int(config.seed),
            "simulations": int(config.simulations),
            "calibration_state_identity": _calibration_state_identity(config),
        },
        "bps_rules": bps.ruleset_fingerprint(),
        "challenger_version": bc.BONUS_BPS_MODEL_VERSION,
    }


def _calibration_state_identity(config: "mc.MonteCarloConfig") -> str:
    """The ACTUAL calibration provenance field, never a silent None."""

    if not hasattr(mc, "calibration_provenance"):
        raise CaptureContractError("monte_carlo.calibration_provenance is unavailable")
    provenance = mc.calibration_provenance(config)
    identity = provenance.get("calibration_state_identity")
    if not identity:
        raise CaptureContractError(
            "calibration provenance carries no calibration_state_identity; refusing to "
            f"emit a silent None (keys: {sorted(provenance)})"
        )
    return str(identity)


def _max_background_event(conn: sqlite3.Connection, *, as_of: str, planning_event: int) -> int | None:
    """Highest event that actually entered the background — must be < planning_event."""

    rows = ho.historical_player_fixtures(conn, as_of=as_of, planning_event=planning_event)
    events = [int(r["fixture_event"]) for r in rows if r.get("fixture_event") is not None]
    return max(events) if events else None


# ---------------------------------------------------------------------------
# Outcomes and metrics
# ---------------------------------------------------------------------------


def final_outcomes(conn: sqlite3.Connection, *, event: int) -> dict[tuple[int, int], dict[str, Any]]:
    """Final official player-fixture outcomes for one FINAL, CHECKED event."""

    rows = conn.execute(
        """
        SELECT pg.player_id AS player_id, pg.fixture_id AS fixture_id, pg.bonus AS bonus,
               pg.minutes AS minutes, pg.bps AS bps
        FROM player_gameweeks pg
        JOIN fixtures f ON f.id = pg.fixture_id
        JOIN events e ON e.id = f.event
        WHERE f.event = ? AND f.finished = 1 AND e.finished = 1 AND e.data_checked = 1
        """,
        (int(event),),
    ).fetchall()
    return {(int(r["player_id"]), int(r["fixture_id"])): dict(r) for r in rows}


def _metric_value(metric: Any) -> Any:
    return getattr(metric, "value", metric)


def score_predictions(predictions: Sequence[PlayerFixturePrediction],
                      outcomes: Mapping[tuple[int, int], Mapping[str, Any]],
                      ) -> dict[str, Any]:
    """PE-2 metric functions over the eligible population.  No duplicate metric maths."""

    eligible: list[tuple[PlayerFixturePrediction, int]] = []
    excluded = {"missing_outcome": 0, "missing_bonus": 0}
    for prediction in predictions:
        outcome = outcomes.get((prediction.player_id, prediction.fixture_id))
        if outcome is None:
            excluded["missing_outcome"] += 1
            continue
        if outcome.get("bonus") is None:
            excluded["missing_bonus"] += 1
            continue
        eligible.append((prediction, int(outcome["bonus"])))

    if not eligible:
        return {"status": "NO_SAMPLE", "n": 0, "excluded": excluded}

    predicted = [p.expected_bonus for p, _ in eligible]
    realised = [float(bonus) for _, bonus in eligible]
    probabilities = [p.p_bonus_any for p, _ in eligible]
    outcomes_any = [1.0 if bonus > 0 else 0.0 for _, bonus in eligible]

    report: dict[str, Any] = {
        "status": "OK",
        "n": len(eligible),
        "mean_predicted_bonus": sum(predicted) / len(predicted),
        "mean_realised_bonus": sum(realised) / len(realised),
        "bias": _metric_value(wfm.mean_bias(predicted, realised)),
        "mae": _metric_value(wfm.mean_absolute_error(predicted, realised)),
        "brier_p_any": _metric_value(wfm.brier_score(probabilities, outcomes_any)),
        "excluded": excluded,
        "by_position": {},
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
            "bias": _metric_value(wfm.mean_bias(gp, gr)),
            "mae": _metric_value(wfm.mean_absolute_error(gp, gr)),
            "brier_p_any": _metric_value(wfm.brier_score(
                [p.p_bonus_any for p, _ in group],
                [1.0 if b > 0 else 0.0 for _, b in group])),
        }
    return report


def soft_baseline(conn: sqlite3.Connection, *, xpts_run_id: int,
                  keys: Iterable[tuple[int, int]],
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
            (int(xpts_run_id), player_id, fixture_id),
        ).fetchone()
        if row is None or row["payload_json"] is None:
            return {"status": "INCOMPLETE_BASELINE", "n": 0,
                    "missing": {"player_id": player_id, "fixture_id": fixture_id}}
        payload = json.loads(row["payload_json"])
        value = payload.get("bonus_xpts")
        if value is None:
            return {"status": "INCOMPLETE_BASELINE", "n": 0,
                    "missing": {"player_id": player_id, "fixture_id": fixture_id}}
        values[(player_id, fixture_id)] = float(value)

    predicted = [values[k] for k in key_list]
    realised = [float(outcomes[k]["bonus"]) for k in key_list]
    return {
        "status": "OK", "n": len(key_list),
        "mean_predicted": sum(predicted) / len(predicted),
        "mean_realised": sum(realised) / len(realised),
        "bias": _metric_value(wfm.mean_bias(predicted, realised)),
        "mae": _metric_value(wfm.mean_absolute_error(predicted, realised)),
        "brier": "NOT_APPLICABLE — expected points is not a probability",
        "population_digest": population_digest(key_list),
    }


def population_digest(keys: Iterable[tuple[int, int]]) -> str:
    payload = ",".join(f"{int(p)}:{int(f)}" for p, f in sorted({(int(a), int(b)) for a, b in keys}))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
