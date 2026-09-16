"""Prospective-analytics spine: projection runs, immutable freezes, baselines.

One projection run (``projection_runs`` row) is one immutable analytics
generation.  Predictions are frozen once per run and can never be mutated
(storage-level triggers).  Baselines are simple, transparent comparators
frozen alongside the model so improvement can be measured, not claimed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from .planning import PlanningContext
from .utils import utc_now
from . import historical_observations as historical
from . import repositories as repo

BASELINE_MODEL_FAMILY = "baseline"
MINUTES_MODEL_FAMILY = "minutes_v1"
TEAM_MODEL_FAMILY = "team_strength_v1"
TEAM_BASELINE_MODEL_FAMILY = "team_baseline"
PLAYER_RATES_MODEL_FAMILY = "player_rates_v1"
PLAYER_RATE_BASELINE_MODEL_FAMILY = "player_rate_baseline"
XPTS_MODEL_FAMILY = "xpts_v1"
# v1.1.0: the naive position-mean-minutes baseline now reads only the causal
# historical window.  Its previous local predicate averaged the 456 scheduled
# placeholders in as real zero-minute appearances, understating every
# positional mean by 4.8-8.9 minutes.  Baseline runs are comparison artifacts
# and are not part of any certified bundle, so no certification identity moves;
# existing baseline runs stay immutable and simply carry the older version.
BASELINE_MODEL_VERSION = "baseline_v1.1.0"

EP_NEXT_KIND = "OFFICIAL_FPL_EP_NEXT"
RECENT_POINTS_KIND = "RECENT_POINTS_BASELINE"
NAIVE_P90_KIND = "NAIVE_P90_BASELINE"
NAIVE_MINUTES_KIND = "NAIVE_MINUTES_BASELINE"
MINUTES_V1_KIND = "MINUTES_V1"

# Recent-evidence window for the transparent baselines (completed league rows).
RECENT_BASELINE_WINDOW = 4
# Minimum completed played minutes before points-per-90 is meaningful.
NAIVE_P90_MINIMUM_PLAYED_MINUTES = 90


def code_revision() -> str | None:
    """Git revision of the running code tree, when available."""

    try:
        import subprocess

        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, ValueError):
        pass
    return None


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


# Source files whose bytes materially determine a certification bundle.  The
# worktree is intentionally dirty/uncommitted, so this reproducible snapshot hash
# is the provenance tie between an artifact and the code that produced it.
SOURCE_SNAPSHOT_FILES = (
    "fpl_brain/analytics.py",
    "fpl_brain/calibration.py",
    "fpl_brain/database.py",
    # The DEFCON probability calibration is part of the certified source set:
    # its versioned spec determines the DEFCON term in every xPts row, so a
    # change to it must change the certified code identity.
    "fpl_brain/defcon_calibration.py",
    "fpl_brain/historical_observations.py",
    # The canonical point-in-time boundary selects which player-fixture
    # observations every historical model may read, so a change to it changes WHAT
    # the certified models consume and must change the certified code identity.
    "fpl_brain/history_completeness.py",
    "fpl_brain/joint_minutes.py",
    "fpl_brain/minutes_coherence.py",
    "fpl_brain/minutes_model.py",
    "fpl_brain/monte_carlo.py",
    "fpl_brain/player_rates.py",
    "fpl_brain/repositories.py",
    "fpl_brain/scoring_rules.py",
    "fpl_brain/substitution_model.py",
    "fpl_brain/team_model.py",
    "fpl_brain/xpts.py",
    # The certification ENTRY POINT is in the identity on purpose: the
    # history-completeness gate lives in its wiring, so a change to that wiring
    # must change the certified code identity rather than pass unnoticed.
    "scripts/certify_gw5_gw8.py",
    "scripts/freeze_predictions.py",
)


def source_snapshot_sha256(paths: Iterable[str] | None = None, root: Path | None = None) -> str:
    """Deterministic hash over sorted ``path + bytes`` of the relevant sources."""

    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in sorted(paths or SOURCE_SNAPSHOT_FILES):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        file_path = base / relative
        if file_path.exists():
            digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    """Branch / HEAD / dirty flag of the running worktree, when available."""

    import subprocess

    info: dict[str, Any] = {"branch": None, "head": None, "dirty": None}
    try:
        head = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if head.returncode == 0:
            info["head"] = head.stdout.strip() or None
        branch = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip() or None
        status = subprocess.run(
            ["git", "-c", "safe.directory=*", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except (OSError, ValueError):
        pass
    return info


def _json_text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def create_projection_run(
    conn: sqlite3.Connection,
    *,
    model_family: str,
    model_version: str,
    planning_event: int,
    planning_context_hash: str | None,
    data_cutoff: str,
    scouting_cutoff: str | None,
    official_run_ids: Mapping[str, Any] | None,
    config_hash: str | None = None,
    random_seed: int | None = None,
    deadline_status: str | None = None,
    source_snapshot_sha256: str | None = None,
    allow_future_cutoff: bool = False,
) -> int:
    # Defense in depth: a run may not claim a data cutoff in the future relative
    # to its own generation.  ``allow_future_cutoff`` exists only for synthetic /
    # historical construction and must never be set by a production path.
    if not allow_future_cutoff:
        from . import causality

        causality.assert_data_cutoff_not_after_generated(data_cutoff, utc_now())
    cursor = conn.execute(
        """INSERT INTO projection_runs(
             model_family, model_version, generated_at, planning_event,
             planning_context_hash, data_cutoff, scouting_cutoff, official_run_ids,
             code_revision, config_hash, random_seed, deadline_status, status,
             source_snapshot_sha256
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'running',?)""",
        (
            model_family,
            model_version,
            utc_now(),
            int(planning_event),
            planning_context_hash,
            data_cutoff,
            scouting_cutoff,
            _json_text_or_none(official_run_ids),
            code_revision(),
            config_hash,
            random_seed,
            deadline_status,
            source_snapshot_sha256,
        ),
    )
    return int(cursor.lastrowid)


def finish_projection_run(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    if status not in {"complete", "failed"}:
        raise ValueError("finish status must be complete or failed")
    conn.execute("UPDATE projection_runs SET status=? WHERE id=?", (status, int(run_id)))


def get_projection_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM projection_runs WHERE id=?", (int(run_id),)).fetchone()
    return dict(row) if row else None


def freeze_prediction(
    conn: sqlite3.Connection,
    projection_run_id: int,
    *,
    kind: str,
    player_id: int,
    event: int,
    payload: Mapping[str, Any],
    fixture_id: int | None = None,
    model_version: str,
    generated_at: str | None = None,
) -> int:
    if kind not in {
        EP_NEXT_KIND,
        RECENT_POINTS_KIND,
        NAIVE_P90_KIND,
        NAIVE_MINUTES_KIND,
        MINUTES_V1_KIND,
    }:
        raise ValueError(f"unsupported frozen prediction kind: {kind}")
    cursor = conn.execute(
        """INSERT INTO frozen_predictions(
             projection_run_id, kind, player_id, fixture_id, event, payload_json,
             model_version, generated_at
           ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            int(projection_run_id),
            kind,
            int(player_id),
            None if fixture_id is None else int(fixture_id),
            int(event),
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            model_version,
            generated_at or utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def frozen_predictions(
    conn: sqlite3.Connection,
    run_id: int,
    kinds: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    clauses = ""
    params: list[Any]
    if kinds is not None:
        kinds_list = sorted(kinds)
        if not kinds_list:
            return []
        params = [int(run_id)] + kinds_list
        clauses = " AND kind IN (" + ",".join("?" for _ in kinds_list) + ")"
    else:
        params = [int(run_id)]
    rows = conn.execute(
        "SELECT * FROM frozen_predictions WHERE projection_run_id=?" + clauses,
        tuple(params),
    ).fetchall()
    records = []
    for row in rows:
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        records.append(record)
    return records


def freeze_team_fixture_projection(
    conn: sqlite3.Connection,
    projection_run_id: int,
    *,
    fixture_id: int,
    event: int,
    team_id: int,
    opponent_id: int,
    venue: str,
    payload: Mapping[str, Any],
    model_version: str,
    generated_at: str | None = None,
) -> int:
    """Append one immutable team-fixture projection against an existing run."""

    if venue not in {"home", "away"}:
        raise ValueError(f"unsupported venue: {venue}")
    cursor = conn.execute(
        """INSERT INTO team_fixture_projections(
             projection_run_id, fixture_id, event, team_id, opponent_id, venue,
             payload_json, model_version, generated_at
           ) VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            int(projection_run_id),
            int(fixture_id),
            int(event),
            int(team_id),
            int(opponent_id),
            venue,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            model_version,
            generated_at or utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def freeze_player_rate_projection(
    conn: sqlite3.Connection,
    projection_run_id: int,
    *,
    player_id: int,
    component: str,
    event: int,
    payload: Mapping[str, Any],
    model_version: str,
    generated_at: str | None = None,
) -> int:
    """Append one immutable player-rate projection against an existing run."""

    cursor = conn.execute(
        """INSERT INTO player_rate_projections(
             projection_run_id, player_id, component, event, payload_json,
             model_version, generated_at
           ) VALUES (?,?,?,?,?,?,?)""",
        (
            int(projection_run_id),
            int(player_id),
            component,
            int(event),
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            model_version,
            generated_at or utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def team_fixture_projections(
    conn: sqlite3.Connection, run_id: int
) -> list[dict[str, Any]]:
    """Frozen team-fixture projections for a run, payload parsed."""

    records = []
    for row in conn.execute(
        "SELECT * FROM team_fixture_projections WHERE projection_run_id=? ORDER BY fixture_id, team_id",
        (int(run_id),),
    ).fetchall():
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        records.append(record)
    return records


def player_rate_projections(
    conn: sqlite3.Connection, run_id: int
) -> list[dict[str, Any]]:
    """Frozen player-rate projections for a run, payload parsed."""

    records = []
    for row in conn.execute(
        "SELECT * FROM player_rate_projections WHERE projection_run_id=? ORDER BY player_id, component",
        (int(run_id),),
    ).fetchall():
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        records.append(record)
    return records


def freeze_xpts_projection(
    conn: sqlite3.Connection,
    projection_run_id: int,
    *,
    player_id: int,
    fixture_id: int,
    event: int,
    team_id: int,
    opponent_id: int,
    position: str,
    minutes_run_id: int,
    team_run_id: int,
    rate_run_id: int,
    payload: Mapping[str, Any],
    model_version: str,
    scoring_rules_version: str,
    generated_at: str | None = None,
) -> int:
    """Append one immutable player-fixture xPts projection against a run."""

    cursor = conn.execute(
        """INSERT INTO player_fixture_xpts_projections(
             projection_run_id, player_id, fixture_id, event, team_id, opponent_id, position,
             minutes_run_id, team_run_id, rate_run_id, payload_json, model_version,
             scoring_rules_version, generated_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            int(projection_run_id),
            int(player_id),
            int(fixture_id),
            int(event),
            int(team_id),
            int(opponent_id),
            position,
            int(minutes_run_id),
            int(team_run_id),
            int(rate_run_id),
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            model_version,
            scoring_rules_version,
            generated_at or utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def xpts_projections(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    """Frozen player-fixture xPts projections for a run, payload parsed."""

    records = []
    for row in conn.execute(
        "SELECT * FROM player_fixture_xpts_projections WHERE projection_run_id=? ORDER BY fixture_id, player_id",
        (int(run_id),),
    ).fetchall():
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        records.append(record)
    return records


def freeze_team_minutes_coherence(
    conn: sqlite3.Connection,
    projection_run_id: int,
    record: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> int:
    """Append one immutable team-coherence record against a minutes run."""

    cursor = conn.execute(
        """INSERT INTO team_minutes_coherence(
             projection_run_id, fixture_id, team_id, players, raw_start_sum, adjusted_start_sum,
             raw_minutes_sum, adjusted_minutes_sum, start_intercept, cameo_intercept,
             start_residual, minutes_residual, status, created_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            int(projection_run_id),
            int(record["fixture_id"]),
            int(record["team_id"]),
            int(record["players"]),
            float(record["raw_start_sum"]),
            float(record["adjusted_start_sum"]),
            float(record["raw_minutes_sum"]),
            float(record["adjusted_minutes_sum"]),
            float(record["start_intercept"]),
            float(record["cameo_intercept"]),
            float(record["start_residual"]),
            float(record["minutes_residual"]),
            str(record["status"]),
            generated_at or utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def freeze_monte_carlo_distribution(
    conn: sqlite3.Connection,
    projection_run_id: int,
    *,
    player_id: int,
    fixture_id: int,
    event: int,
    team_id: int,
    opponent_id: int,
    position: str,
    xpts_run_id: int,
    minutes_run_id: int,
    team_run_id: int,
    rate_run_id: int,
    payload: Mapping[str, Any],
    model_version: str,
    generated_at: str | None = None,
) -> int:
    """Append one immutable Monte Carlo distribution summary."""

    cursor = conn.execute(
        """INSERT INTO monte_carlo_distributions(
             projection_run_id, player_id, fixture_id, event, team_id, opponent_id, position,
             xpts_run_id, minutes_run_id, team_run_id, rate_run_id, payload_json,
             model_version, generated_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            int(projection_run_id), int(player_id), int(fixture_id), int(event), int(team_id),
            int(opponent_id), position, int(xpts_run_id), int(minutes_run_id), int(team_run_id),
            int(rate_run_id), json.dumps(payload, ensure_ascii=False, sort_keys=True), model_version,
            generated_at or utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def monte_carlo_distributions(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    """Frozen Monte Carlo distribution summaries for a run, payload parsed."""

    records = []
    for row in conn.execute(
        "SELECT * FROM monte_carlo_distributions WHERE projection_run_id=? ORDER BY fixture_id, player_id",
        (int(run_id),),
    ).fetchall():
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        records.append(record)
    return records


def freeze_team_substitution_profile(
    conn: sqlite3.Connection,
    projection_run_id: int,
    profile: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> int:
    """Append one immutable team substitution profile against a minutes run."""

    cursor = conn.execute(
        """INSERT INTO team_substitution_profiles(
             projection_run_id, fixture_id, team_id, expected_substitutions, expected_exit_mass,
             expected_entry_mass, gk_event_mass, evidence_matches, prior_source, status,
             payload_json, created_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            int(projection_run_id), int(profile["fixture_id"]), int(profile["team_id"]),
            float(profile["expected_substitutions"]), float(profile["expected_exit_mass"]),
            float(profile["expected_entry_mass"]), float(profile.get("gk_event_mass") or 0.0),
            int(profile.get("evidence_matches") or 0), profile.get("prior_source"),
            str(profile.get("status") or "COHERENT"),
            json.dumps(profile, ensure_ascii=False, sort_keys=True), generated_at or utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def team_substitution_profiles(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    """Frozen team substitution profiles for a run, payload parsed."""

    records = []
    for row in conn.execute(
        "SELECT * FROM team_substitution_profiles WHERE projection_run_id=? ORDER BY fixture_id, team_id",
        (int(run_id),),
    ).fetchall():
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        records.append(record)
    return records


def team_minutes_coherence_records(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    """Frozen team-coherence records for a minutes run."""

    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM team_minutes_coherence WHERE projection_run_id=? ORDER BY fixture_id, team_id",
            (int(run_id),),
        ).fetchall()
    ]


def projectable_players(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every active player with an official club; blank-week players simply
    produce no fixture-level predictions."""

    return [
        dict(row)
        for row in conn.execute(
            """SELECT p.id AS player_id, p.team_id, p.element_type, p.is_active
               FROM players p
                WHERE p.is_active=1 AND p.team_id IS NOT NULL
                ORDER BY p.id"""
        ).fetchall()
    ]


def event_fixture_map(conn: sqlite3.Connection, event: int) -> dict[int, list[dict[str, Any]]]:
    """Team id -> fixtures of the event (each row keeps its fixture id)."""

    mapping: dict[int, list[dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT * FROM fixtures WHERE event=? ORDER BY id", (int(event),)
    ).fetchall():
        record = dict(row)
        for side in ("team_h", "team_a"):
            team = record.get(side)
            if team is not None:
                mapping.setdefault(int(team), []).append(record)
    return mapping


def snapshot_as_of(conn: sqlite3.Connection, player_id: int, cutoff: str) -> dict[str, Any] | None:
    """Freshest official snapshot captured at or before the cutoff."""

    row = conn.execute(
        """SELECT * FROM player_snapshots WHERE player_id=? AND captured_at<=?
           ORDER BY captured_at DESC, id DESC LIMIT 1""",
        (int(player_id), cutoff),
    ).fetchone()
    return dict(row) if row else None


def snapshot_history_as_of(
    conn: sqlite3.Connection, player_id: int, cutoff: str
) -> list[dict[str, Any]]:
    """The official status trail at or before the cutoff, oldest first.

    Used to recover the availability that applied at a *historical* fixture so a
    0-minute row can be told apart from an unavailable player.  No lookahead:
    only snapshots captured at or before the cutoff are returned.
    """

    rows = conn.execute(
        """SELECT * FROM player_snapshots WHERE player_id=? AND captured_at<=?
           ORDER BY captured_at ASC, id ASC""",
        (int(player_id), cutoff),
    ).fetchall()
    return [dict(row) for row in rows]


def completed_rows_as_of(
    conn: sqlite3.Connection,
    player_id: int,
    cutoff: str,
    planning_event: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Most recent completed league player-fixture rows before the cutoff.

    No-lookahead core: the fixture must be strictly earlier than the planning
    event and its kickoff must not be after the cutoff, so post-deadline facts
    can never leak into a freeze.
    """

    rows = historical.historical_player_fixture_rows(
        conn, as_of=str(cutoff), planning_event=int(planning_event),
        player_id=int(player_id), limit=limit,
    )
    for record in rows:
        # The observation universe now EXCLUDES stale schedule placeholders at the
        # canonical boundary, so a returned row is a legitimate realised
        # observation by construction.  The flag is retained for diagnostics and
        # for consumers that still read it, but it is no longer the only thing
        # standing between a model and placeholder history.
        record["history_placeholder"] = False
    return rows


def snapshot_status_evidence(conn: sqlite3.Connection, player_id: int, cutoff: str) -> dict[str, Any]:
    """Official availability evidence from the freshest pre-cutoff snapshot."""

    snapshot = snapshot_as_of(conn, int(player_id), cutoff)
    return {
        "snapshot_as_of": (snapshot or {}).get("captured_at"),
        "status": (snapshot or {}).get("status"),
        "news": (snapshot or {}).get("news"),
        "chance_of_playing_this_round": (snapshot or {}).get("chance_of_playing_this_round"),
    }


def recent_points_baseline(conn: sqlite3.Connection, player_id: int, planning_event: int, cutoff: str) -> dict[str, Any]:
    """Mean FPL points over the last up to 4 completed league rows.

    A 0-minute row counts as points 0; missing point rows are a DATA GAP,
    never fabricated.
    """

    rows = completed_rows_as_of(conn, int(player_id), cutoff, int(planning_event), RECENT_BASELINE_WINDOW)
    values = [row.get("total_points") for row in rows]
    missing = [index for index, value in enumerate(values) if value is None]
    sampled = [value for value in values if value is not None]
    gaps: list[str] = []
    value = None
    if not rows:
        gaps.append("no completed league rows before cutoff; recent-points baseline unavailable")
    elif missing:
        gaps.append(f"{len(missing)} recent rows missing official total points")
    if sampled:
        value = round(sum(sampled) / len(sampled), 6)
    return {
        "value": value,
        "sample_rows": len(rows),
        "window_rows": [int(row["fixture_id"]) for row in rows],
        "formula": "mean(total_points) over last <= 4 completed league player-fixture rows before cutoff",
        "data_gaps": gaps,
    }


def naive_minutes_baseline(
    conn: sqlite3.Connection,
    player_id: int,
    planning_event: int,
    cutoff: str,
    pooled_prior: float | None = None,
) -> dict[str, Any]:
    """Average minutes over the last up to 4 completed league rows.

    Transparent comparator; the fallback prior is the positional pooled mean
    minutes from the same completed window.
    """

    rows = completed_rows_as_of(conn, int(player_id), cutoff, int(planning_event), RECENT_BASELINE_WINDOW)
    minutes = [row.get("minutes") for row in rows]
    starts = [row.get("starts") for row in rows]
    sampled = [value for value in minutes if value is not None]
    starts_known = [value for value in starts if value is not None]
    gaps: list[str] = []
    if sampled:
        value = round(sum(sampled) / len(sampled), 6)
    elif pooled_prior is not None:
        value = round(float(pooled_prior), 6)
        gaps.append("no completed league rows for the player; positional pooled mean minutes used")
    else:
        value = None
        gaps.append("no completed league rows and no pooled fallback available")
    start_share = (
        round(sum(1 for v in starts_known if v) / len(starts_known), 6) if starts_known else None
    )
    return {
        "value": value,
        "sample_rows": len(rows),
        "window_rows": [int(row["fixture_id"]) for row in rows],
        "p_start": start_share,
        "formula": "mean(minutes) over last <= 4 completed league player-fixture rows before cutoff; fallback = positional pooled mean minutes",
        "data_gaps": gaps,
    }


def recent_points_per_90(
    conn: sqlite3.Connection,
    player_id: int,
    planning_event: int,
    cutoff: str,
    window_rows: int = RECENT_BASELINE_WINDOW,
) -> dict[str, Any]:
    """Transparent points-per-90 over the recent window (played rows only)."""

    rows = completed_rows_as_of(conn, int(player_id), cutoff, int(planning_event), window_rows)
    played = [row for row in rows if (row.get("minutes") or 0) > 0]
    minutes = int(sum(row.get("minutes") or 0 for row in played))
    points = int(sum(row.get("total_points") or 0 for row in played))
    if minutes < NAIVE_P90_MINIMUM_PLAYED_MINUTES:
        return {
            "per_90": None,
            "played_rows": len(played),
            "minutes": minutes,
            "points": points,
            "data_gaps": [
                f"only {minutes} played minutes recorded; below the {NAIVE_P90_MINIMUM_PLAYED_MINUTES}-minute minimum for a stable per-90"
            ],
        }
    return {
        "per_90": round(points / minutes * 90.0, 6),
        "played_rows": len(played),
        "minutes": minutes,
        "points": points,
        "data_gaps": [],
    }


def naive_p90_baseline(
    conn: sqlite3.Connection,
    player_id: int,
    planning_event: int,
    cutoff: str,
    expected_minutes: float | None,
) -> dict[str, Any]:
    """Historical/current points per 90 times a naive minutes expectation."""

    decomposition = recent_points_per_90(conn, int(player_id), planning_event, cutoff)
    per_90 = decomposition["per_90"]
    value = None
    gaps = list(decomposition["data_gaps"])
    if per_90 is not None and expected_minutes is not None:
        value = round(float(per_90) * float(expected_minutes) / 90.0, 6)
    elif per_90 is not None and expected_minutes is None:
        gaps.append("naive minutes expectation unavailable")
    return {
        "value": value,
        "per_90": per_90,
        "expected_minutes": expected_minutes,
        "formula": "points_per_90(recent played rows, min 90') x naive_expected_minutes / 90",
        "played_rows": decomposition["played_rows"],
        "data_gaps": gaps,
    }


def positional_pooled_minutes(conn: sqlite3.Connection, planning_event: int, cutoff: str) -> dict[int, float]:
    """Position-pooled mean minutes over the causally available window.

    A genuine zero-minute non-appearance is an observation and is retained; a
    scheduled placeholder is not and is excluded at the canonical boundary.
    This reader previously applied the local predicate directly, so the 456
    placeholders stored for a finished-but-unreported fixture were averaged in
    as real zero-minute appearances and dragged every positional mean down.
    """

    pooled: dict[int, float] = {}
    for row in conn.execute(
        f"""SELECT p.element_type AS pos, AVG(pg.minutes) AS avg_minutes
           FROM player_gameweeks pg
           JOIN players p ON p.id=pg.player_id
           JOIN fixtures f ON f.id=pg.fixture_id
          WHERE {historical.OBSERVATION_SQL_CLAUSES}
          GROUP BY p.element_type""",
        historical.boundary_params(cutoff, planning_event=int(planning_event)),
    ).fetchall():
        if row["pos"] is not None and row["avg_minutes"] is not None:
            pooled[int(row["pos"])] = round(float(row["avg_minutes"]), 6)
    return pooled


def positional_pooled_engagement(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
) -> dict[int, dict[str, float]]:
    """Position-pooled start/minutes/tail rates over completed rows.

    These are the transparent level-4 priors of the minutes hierarchy (start
    rate, mean minutes given start, share of starts reaching 60/80 minutes,
    cameo rate among not-starting rows and their mean minutes).  They come
    from the causally available league window before the cutoff with no player
    identity involved.

    The window is the canonical historical boundary, not a local predicate: a
    row the league had not written when the cutoff passed is not evidence, and
    a row whose own fixture had not kicked off is a scheduled placeholder
    rather than an observation.  ``starts IS NOT NULL AND minutes IS NOT NULL``
    is this reader's own modelling filter on top of that window.
    """

    fallback: dict[int, dict[str, float]] = {}
    rows = conn.execute(
        f"""SELECT p.element_type AS pos, pg.starts AS started, pg.minutes AS minutes
           FROM player_gameweeks pg
           JOIN players p ON p.id=pg.player_id
           JOIN fixtures f ON f.id=pg.fixture_id
          WHERE {historical.OBSERVATION_SQL_CLAUSES}
            AND pg.starts IS NOT NULL AND pg.minutes IS NOT NULL""",
        historical.boundary_params(cutoff, planning_event=int(planning_event)),
    ).fetchall()
    pools: dict[int, dict[str, float]] = {}
    for row in rows:
        pos = row["pos"]
        if pos is None:
            continue
        pool = pools.setdefault(
            int(pos), {"starts": 0.0, "rows": 0.0, "start_minutes": 0.0, "p60": 0.0, "p80": 0.0, "cameo_rows": 0.0, "cameo_minutes_total": 0.0, "cameo60": 0.0}
        )
        minutes = float(row["minutes"])
        if row["started"]:
            pool["rows"] += 1
            pool["starts"] += 1
            pool["start_minutes"] += minutes
            pool["p60"] += 1 if minutes >= 60 else 0
            pool["p80"] += 1 if minutes >= 80 else 0
        else:
            pool["rows"] += 1
            if minutes > 0:
                pool["cameo_rows"] += 1
                pool["cameo_minutes_total"] += minutes
                pool["cameo60"] += 1 if minutes >= 60 else 0

    result: dict[int, dict[str, float]] = {}
    for pos, pool in pools.items():
        starts = float(pool["starts"])
        rows_count = float(pool["rows"])
        not_start_rows = rows_count - starts
        cameo_rows = float(pool["cameo_rows"])
        cameo_minutes = pool["cameo_minutes_total"] / cameo_rows if cameo_rows else 15.0
        result[pos] = {
            "start_rows": int(rows_count),
            "p_start": round(starts / rows_count, 6) if rows_count else 0.0,
            "minutes_if_start": round(pool["start_minutes"] / starts, 6) if starts else 0.0,
            "p60_if_start": round(pool["p60"] / starts, 6) if starts else 0.0,
            "p80_if_start": round(pool["p80"] / starts, 6) if starts else 0.0,
            "not_start_rows": int(not_start_rows),
            "cameo_rate": round(cameo_rows / not_start_rows, 6) if not_start_rows else 0.0,
            "cameo_rows": int(cameo_rows),
            "cameo_minutes": round(cameo_minutes, 6),
            "cameo_p60": round(pool["cameo60"] / cameo_rows, 6) if cameo_rows else 0.0,
        }
    return result


def ep_next_as_of(conn: sqlite3.Connection, player_id: int, cutoff: str) -> Any:
    snapshot = snapshot_as_of(conn, int(player_id), cutoff)
    if not snapshot:
        return None
    value = snapshot.get("ep_next")
    if value is None:
        value = snapshot.get("ep_this")
    return value


def planning_context_reference(context: PlanningContext) -> str:
    """Stable reference of the frozen PlanningContext subset used by runs."""

    official_runs = context.official_runs or {}

    def nested(key: str) -> dict[str, Any]:
        return dict(official_runs.get(key) or {})

    return canonical_hash(
        {
            "entry_id": context.entry_id,
            "season": context.season,
            "planning_event": context.planning_event,
            "as_of": context.as_of,
            "event_data_state": context.event_data_state,
            "deadline": context.deadline,
            "fetch_run": nested("fetch").get("run_id"),
            "manager_sync_run": nested("manager_sync").get("run_id"),
            "scouting_cutoff": context.scouting_cutoff,
            "manager_state": {
                "free_transfers": (context.manager_state or {}).get("free_transfers"),
                "bank": (context.manager_state or {}).get("bank"),
                "free_transfers_source": (context.manager_state or {}).get("free_transfers_source"),
                "bank_source": (context.manager_state or {}).get("bank_source"),
            },
            "squad": {
                "squad_state": (context.squad or {}).get("squad_state"),
                "player_count": (context.squad or {}).get("player_count"),
            },
            "official_price_freshness": (context.official_price_freshness or {}).get("freshness"),
            "health_status": (context.health or {}).get("status"),
        }
    )


def freeze_baselines_for_event(
    conn: sqlite3.Connection,
    context: PlanningContext,
    cutoff: str,
    *,
    event: int | None = None,
    deadline_status: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Freeze the four transparent baselines for every projectable player.

    Baselines are exact at freeze time and never recomputed later.
    """

    planning_event = int(event if event is not None else context.planning_event)
    run_id = create_projection_run(
        conn,
        model_family=BASELINE_MODEL_FAMILY,
        model_version=BASELINE_MODEL_VERSION,
        planning_event=planning_event,
        planning_context_hash=planning_context_reference(context),
        data_cutoff=cutoff,
        scouting_cutoff=context.scouting_cutoff,
        official_run_ids=context.official_runs,
        config_hash=canonical_hash(
            {"window": RECENT_BASELINE_WINDOW, "p90_min_minutes": NAIVE_P90_MINIMUM_PLAYED_MINUTES}
        ),
        deadline_status=deadline_status,
    )
    fixtures_by_team = event_fixture_map(conn, planning_event)
    pooled_minutes = positional_pooled_minutes(conn, planning_event, cutoff)
    count_rows = {"ep": 0, "recent": 0, "minutes": 0, "p90": 0}
    for player in projectable_players(conn):
        player_id = int(player["player_id"])
        if not fixtures_by_team.get(int(player["team_id"])):
            continue
        official = snapshot_status_evidence(conn, player_id, cutoff)
        freeze_prediction(
            conn,
            run_id,
            kind=EP_NEXT_KIND,
            player_id=player_id,
            event=planning_event,
            payload={
                "value": ep_next_as_of(conn, player_id, cutoff),
                "provenance": official,
                "formula": "official FPL ep_next captured at freeze time, never recomputed",
            },
            model_version=BASELINE_MODEL_VERSION,
        )
        count_rows["ep"] += 1
        freeze_prediction(
            conn,
            run_id,
            kind=RECENT_POINTS_KIND,
            player_id=player_id,
            event=planning_event,
            payload=recent_points_baseline(conn, player_id, planning_event, cutoff),
            model_version=BASELINE_MODEL_VERSION,
        )
        count_rows["recent"] += 1
        naive_minutes = naive_minutes_baseline(
            conn,
            player_id,
            planning_event,
            cutoff,
            pooled_prior=pooled_minutes.get(int(player["element_type"])),
        )
        freeze_prediction(
            conn,
            run_id,
            kind=NAIVE_MINUTES_KIND,
            player_id=player_id,
            event=planning_event,
            payload={**naive_minutes, "pooled_minutes_prior": pooled_minutes.get(int(player["element_type"]))},
            model_version=BASELINE_MODEL_VERSION,
        )
        count_rows["minutes"] += 1
        freeze_prediction(
            conn,
            run_id,
            kind=NAIVE_P90_KIND,
            player_id=player_id,
            event=planning_event,
            payload=naive_p90_baseline(conn, player_id, planning_event, cutoff, naive_minutes.get("value")),
            model_version=BASELINE_MODEL_VERSION,
        )
        count_rows["p90"] += 1
    finish_projection_run(conn, run_id, "complete")
    return run_id, {"run_id": run_id, **count_rows}
