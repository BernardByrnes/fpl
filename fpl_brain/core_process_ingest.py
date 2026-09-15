"""FPL Core Insights ingestion — source contract, immutable cache, crosswalks.

SOURCE CONTRACT
---------------
Primary external source: ``olbauday/FPL-Core-Insights``.  Only PREMIER LEAGUE
tournament data may reach the predictive layer, so the tournament must be pinned
per file: the repository's own tournament CODE (``prem``) is checked, not only the
directory name, because the same competition can appear under several directory
layouts and other competitions carry their own codes.

Source classes are explicit:

* ``OFFICIAL_FPL`` — authoritative for FPL identity, team/position,
  minutes/starts, goals/assists, FPL points, bonus/BPS, official xG/xA/xGI,
  defensive contribution, fixtures and state;
* ``EXTERNAL_FPL_CORE_INSIGHTS`` — process data only; it SUPPLEMENTS and may
  never silently overwrite an official value;
* ``BRAIN_DERIVED`` — anything Brain computes from the above;
* ``MODEL`` — fitted outputs.

IMMUTABILITY AND POINT-IN-TIME SAFETY
-------------------------------------
Every ingested file is copied byte-for-byte into a commit-scoped cache and
recorded with its hash, row count and schema hash.  A provider correction
allocates a NEW version rather than rewriting history, and availability is the
RETRIEVAL time — never the match kickoff, the match finish, or a Git author time.
Post-match process data from a fixture therefore cannot leak into a decision made
before that fixture, however the cache is read back.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .utils import utc_now

SOURCE_REPOSITORY = "https://github.com/olbauday/FPL-Core-Insights"
SOURCE_CLASS_OFFICIAL_FPL = "OFFICIAL_FPL"
SOURCE_CLASS_EXTERNAL = "EXTERNAL_FPL_CORE_INSIGHTS"
SOURCE_CLASS_BRAIN_DERIVED = "BRAIN_DERIVED"
SOURCE_CLASS_MODEL = "MODEL"
SOURCE_CLASSES = (
    SOURCE_CLASS_OFFICIAL_FPL,
    SOURCE_CLASS_EXTERNAL,
    SOURCE_CLASS_BRAIN_DERIVED,
    SOURCE_CLASS_MODEL,
)

#: The tournament code used by the source for the Premier League.  Directories
#: are labelled "Premier League"; the rows carry "prem".  Both are checked.
PREMIER_LEAGUE_TOURNAMENT = "Premier League"
PREMIER_LEAGUE_TOURNAMENT_CODE = "prem"

#: Files the process layer requires for one gameweek.
REQUIRED_SOURCE_FILES = (
    "players.csv",
    "matches.csv",
    "playermatchstats.csv",
    "shots.csv",
    "player_gameweek_stats.csv",
)

CORPORA = {
    "2026-2027": "By Tournament",
    "2025-2026": "By Tournament",
    "2024-2025": "By Tournament",
}

CORE_PROCESS_SCHEMA = "fpl_brain.core_process_ingest.v1"
CACHE_MANIFEST_NAME = "manifest.json"

DIAG_SOURCE_FILE_MISSING = "CORE_SOURCE_FILE_MISSING"
DIAG_SOURCE_TREE_UNAVAILABLE = "CORE_SOURCE_TREE_UNAVAILABLE"
DIAG_TOURNAMENT_LEAK = "CORE_TOURNAMENT_LEAK"
DIAG_RAW_REWRITE = "CORE_RAW_SNAPSHOT_REWRITE_ATTEMPT"
DIAG_AMBIGUOUS_PLAYER = "CORE_AMBIGUOUS_PLAYER_IDENTITY"
DIAG_PLAYER_UNMATCHED = "CORE_PLAYER_IDENTITY_UNMATCHED"
DIAG_AMBIGUOUS_FIXTURE = "CORE_AMBIGUOUS_FIXTURE_MAPPING"
DIAG_FIXTURE_UNMATCHED = "CORE_FIXTURE_UNMATCHED"
DIAG_SNAPSHOT_NOT_AVAILABLE = "CORE_SNAPSHOT_NOT_AVAILABLE_AT_CUTOFF"
DIAG_ZERO_MINUTE_PLACEHOLDER = "CORE_ZERO_MINUTE_PLACEHOLDER"


class CoreProcessError(ValueError):
    """The source contract was violated; ingestion fails closed."""

    def __init__(self, detail: str, *, reasons: Sequence[str] = ()) -> None:
        super().__init__(detail)
        self.reasons = list(reasons)


# ---------------------------------------------------------------------------
# reading the raw source
# ---------------------------------------------------------------------------


@runtime_checkable
class SourceTree(Protocol):
    """A read-only view of the external repository at a pinned commit."""

    def exists(self, relative_path: str) -> bool:
        ...

    def read_bytes(self, relative_path: str) -> bytes:
        ...


@dataclass(frozen=True)
class LocalSourceTree:
    """A checked-out/extracted copy of the source repository."""

    root: Path
    commit_sha: str

    def _path(self, relative_path: str) -> Path:
        return Path(self.root) / relative_path

    def exists(self, relative_path: str) -> bool:
        return self._path(relative_path).is_file()

    def read_bytes(self, relative_path: str) -> bytes:
        path = self._path(relative_path)
        if not path.is_file():
            raise CoreProcessError(
                f"{DIAG_SOURCE_FILE_MISSING}: {relative_path}",
                reasons=[DIAG_SOURCE_FILE_MISSING],
            )
        return path.read_bytes()


def gameweek_path(*, season: str, tournament: str, gameweek: int, name: str) -> str:
    """The canonical by-tournament path for one source file."""

    return f"data/{season}/By Tournament/{tournament}/GW{int(gameweek)}/{name}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def schema_hash(payload: bytes) -> str:
    """Hash of the file's header row — the column contract, not the data."""

    text = payload.decode("utf-8-sig", errors="replace")
    header = text.splitlines()[0] if text.splitlines() else ""
    return hashlib.sha256(header.encode("utf-8")).hexdigest()


def parse_csv(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


# ---------------------------------------------------------------------------
# the immutable cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFileRecord:
    """Provenance for one cached source file.  Append-only; never edited."""

    source_repository: str
    repository_commit_sha: str
    season: str
    gameweek: int
    tournament: str
    tournament_code: str
    source_path: str
    retrieved_at: str
    sha256: str
    size_bytes: int
    row_count: int
    schema_hash: str
    cache_path: str
    version: int = 1
    source_class: str = SOURCE_CLASS_EXTERNAL
    schema: str = CORE_PROCESS_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_repository": self.source_repository,
            "repository_commit_sha": self.repository_commit_sha,
            "source_class": self.source_class,
            "season": self.season,
            "gameweek": int(self.gameweek),
            "tournament": self.tournament,
            "tournament_code": self.tournament_code,
            "source_path": self.source_path,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
            "row_count": int(self.row_count),
            "schema_hash": self.schema_hash,
            "cache_path": self.cache_path,
            "version": int(self.version),
        }

    def available_at(self, cutoff: str | None) -> bool:
        """Point-in-time rule: RETRIEVAL time, nothing else, decides availability."""

        if cutoff is None:
            return True
        return str(self.retrieved_at) <= str(cutoff)


class CoreProcessCache:
    """Byte-preserving, versioned cache of the external source.

    Re-caching identical bytes is idempotent; re-caching DIFFERENT bytes for the
    same logical file allocates the next version and appends a new manifest entry,
    so a provider correction can never rewrite historical evidence in place.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        repository_commit_sha: str,
        source_repository: str = SOURCE_REPOSITORY,
        retrieved_at: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.repository_commit_sha = str(repository_commit_sha)
        self.source_repository = source_repository
        self.retrieved_at = str(retrieved_at) if retrieved_at is not None else utc_now()
        self.records: list[SourceFileRecord] = []

    # -- paths ---------------------------------------------------------------

    def _dir(self, *, season: str, tournament: str, gameweek: int, version: int) -> Path:
        return (
            self.root
            / self.repository_commit_sha
            / season
            / tournament
            / f"GW{int(gameweek)}"
            / f"v{int(version)}"
        )

    def manifest_path(self) -> Path:
        return self.root / self.repository_commit_sha / CACHE_MANIFEST_NAME

    # -- writing -------------------------------------------------------------

    def cache_file(
        self,
        tree: SourceTree,
        *,
        season: str,
        gameweek: int,
        name: str,
        tournament: str = PREMIER_LEAGUE_TOURNAMENT,
        expected_tournament_code: str = PREMIER_LEAGUE_TOURNAMENT_CODE,
        require_tournament: bool = True,
    ) -> SourceFileRecord:
        """Cache one source file, proving its tournament before anything else."""

        source_path = gameweek_path(season=season, tournament=tournament, gameweek=gameweek, name=name)
        payload = tree.read_bytes(source_path)
        rows = parse_csv(payload)
        if require_tournament:
            self.assert_premier_league(rows, source_path=source_path,
                                       expected_code=expected_tournament_code)
        digest = sha256_bytes(payload)
        version = self._version_for(season=season, tournament=tournament, gameweek=gameweek, name=name, digest=digest)
        target_dir = self._dir(season=season, tournament=tournament, gameweek=gameweek, version=version)
        target = target_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_bytes(target.read_bytes()) != digest:
            # Never overwrite: this branch is unreachable by construction because
            # the version was allocated from the existing bytes, and it exists only
            # to fail loudly if that invariant is ever broken.
            raise CoreProcessError(
                f"{DIAG_RAW_REWRITE}: refusing to overwrite {target} with different bytes",
                reasons=[DIAG_RAW_REWRITE],
            )
        if not target.exists():
            target.write_bytes(payload)
        record = SourceFileRecord(
            source_repository=self.source_repository,
            repository_commit_sha=self.repository_commit_sha,
            season=str(season),
            gameweek=int(gameweek),
            tournament=str(tournament),
            tournament_code=str(expected_tournament_code),
            source_path=source_path,
            retrieved_at=self.retrieved_at,
            sha256=digest,
            size_bytes=len(payload),
            row_count=len(rows),
            schema_hash=schema_hash(payload),
            cache_path=str(target),
            version=int(version),
        )
        self.records.append(record)
        return record

    def _version_for(self, *, season: str, tournament: str, gameweek: int, name: str, digest: str) -> int:
        """Version 1 unless a DIFFERENT payload is already cached for this file."""

        for record in self.manifest():
            if (
                record["season"] == str(season)
                and record["tournament"] == str(tournament)
                and int(record["gameweek"]) == int(gameweek)
                and Path(record["source_path"]).name == name
            ):
                if record["sha256"] == digest:
                    return int(record["version"])
                return int(record["version"]) + 1
        return 1

    def write_manifest(self) -> Path:
        """Append this run's records to the commit's manifest (never rewrites)."""

        path = self.manifest_path()
        existing = self.manifest()
        known = {(r["source_path"], r["sha256"]) for r in existing}
        merged = existing + [r.as_dict() for r in self.records if (r.source_path, r.sha256) not in known]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def manifest(self) -> list[dict[str, Any]]:
        path = self.manifest_path()
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # -- reading -------------------------------------------------------------

    def available_records(
        self, *, season: str, gameweek: int | None = None, cutoff: str | None = None
    ) -> list[dict[str, Any]]:
        """Manifest entries for a season, filtered by the POINT-IN-TIME cutoff."""

        out = []
        for record in self.manifest():
            if record["season"] != str(season):
                continue
            if gameweek is not None and int(record["gameweek"]) != int(gameweek):
                continue
            if cutoff is not None and str(record["retrieved_at"]) > str(cutoff):
                continue
            out.append(record)
        return sorted(out, key=lambda r: (int(r["gameweek"]), r["source_path"], int(r["version"])))

    # -- tournament guard ----------------------------------------------------

    @staticmethod
    def assert_premier_league(
        rows: Sequence[Mapping[str, Any]], *, source_path: str, expected_code: str = PREMIER_LEAGUE_TOURNAMENT_CODE
    ) -> None:
        """Fail closed when a file carries any non-Premier-League fixture row."""

        if not rows:
            return
        if "tournament" not in rows[0]:
            return  # per-player files carry no tournament column; the path pins it
        codes = sorted({str(row.get("tournament") or "") for row in rows})
        if codes != [str(expected_code)]:
            raise CoreProcessError(
                f"{DIAG_TOURNAMENT_LEAK}: {source_path} carries tournament codes {codes}, "
                f"expected only {expected_code!r}",
                reasons=[DIAG_TOURNAMENT_LEAK],
            )


# ---------------------------------------------------------------------------
# player identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerCrosswalkEntry:
    official_id: int
    external_player_id: int
    external_player_code: int
    web_name: str
    matched_by: str


@dataclass(frozen=True)
class PlayerCrosswalk:
    """Deterministic official <-> external player identity.

    ``player_id`` is matched against the official element id and ``player_code``
    against the official element code; both are checked and must agree.  Names are
    reported for diagnosis only and never used as the production join.
    """

    entries: tuple[PlayerCrosswalkEntry, ...]
    official_active: int
    external_count: int
    id_matches: int
    code_matches: int
    both_keys_agree: int
    ambiguous: tuple[int, ...]
    external_only: tuple[int, ...]
    official_only: tuple[int, ...]
    name_only_matches: int

    @property
    def ok(self) -> bool:
        return not self.ambiguous and not self.official_only

    def official_id_for(self, external_player_id: int) -> int | None:
        for entry in self.entries:
            if entry.external_player_id == int(external_player_id):
                return int(entry.official_id)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "official_active": int(self.official_active),
            "external_count": int(self.external_count),
            "id_matches": int(self.id_matches),
            "code_matches": int(self.code_matches),
            "both_keys_agree": int(self.both_keys_agree),
            "ambiguous": list(self.ambiguous),
            "external_only": list(self.external_only),
            "official_only": list(self.official_only),
            "name_only_matches": int(self.name_only_matches),
            "ok": bool(self.ok),
        }


def _normalise_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def build_player_crosswalk(
    official_players: Sequence[Mapping[str, Any]], external_players: Sequence[Mapping[str, Any]]
) -> PlayerCrosswalk:
    """Match on ids, prove the two keys agree, and report every disagreement."""

    by_id = {int(row["id"]): row for row in official_players if row.get("id") is not None}
    by_code = {
        int(row["code"]): int(row["id"])
        for row in official_players
        if row.get("code") is not None and row.get("id") is not None
    }
    entries: list[PlayerCrosswalkEntry] = []
    ambiguous: list[int] = []
    external_only: list[int] = []
    id_matches = code_matches = agree = name_only = 0
    seen_official: set[int] = set()

    for row in sorted(external_players, key=lambda r: int(r["player_id"])):
        external_id = int(row["player_id"])
        external_code = int(row["player_code"]) if row.get("player_code") not in (None, "") else None
        official_by_id = by_id.get(external_id)
        official_by_code = by_code.get(int(external_code)) if external_code is not None else None
        if official_by_id is not None:
            id_matches += 1
        if official_by_code is not None:
            code_matches += 1
        candidates = {c for c in (official_by_id and int(official_by_id["id"]), official_by_code) if c is not None}
        if len(candidates) > 1:
            ambiguous.append(external_id)
            continue
        if not candidates:
            external_only.append(external_id)
            continue
        official_id = candidates.pop()
        if official_by_id is not None and official_by_code is not None:
            agree += 1
        matched_by = "player_id+player_code" if (official_by_id is not None and official_by_code is not None) else (
            "player_id" if official_by_id is not None else "player_code"
        )
        entries.append(
            PlayerCrosswalkEntry(
                official_id=int(official_id),
                external_player_id=int(external_id),
                external_player_code=int(external_code) if external_code is not None else -1,
                web_name=str(row.get("web_name") or ""),
                matched_by=matched_by,
            )
        )
        seen_official.add(int(official_id))

    # diagnostic-only: how well names would have done
    official_names = {_normalise_name(row.get("web_name")): int(row["id"]) for row in official_players
                      if row.get("id") is not None}
    for row in external_players:
        if _normalise_name(row.get("web_name")) in official_names:
            name_only += 1

    official_only = sorted(set(by_id) - seen_official)
    return PlayerCrosswalk(
        entries=tuple(entries),
        official_active=len(by_id),
        external_count=len(external_players),
        id_matches=id_matches,
        code_matches=code_matches,
        both_keys_agree=agree,
        ambiguous=tuple(sorted(ambiguous)),
        external_only=tuple(sorted(external_only)),
        official_only=tuple(official_only),
        name_only_matches=name_only,
    )


# ---------------------------------------------------------------------------
# fixture identity
# ---------------------------------------------------------------------------

_KICKOFF_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def normalise_kickoff(value: Any) -> str:
    """Second-resolution UTC instant, so 'Z' and a bare naive stamp compare equal."""

    match = _KICKOFF_RE.match(str(value or "").strip())
    if not match:
        raise CoreProcessError(f"unparseable kickoff {value!r}")
    return match.group(1)


def _team_code(value: Any) -> int:
    return int(float(str(value).strip()))


@dataclass(frozen=True)
class FixtureCrosswalkEntry:
    external_match_id: str
    official_fixture_id: int
    event: int
    kickoff: str
    home_team_id: int
    away_team_id: int
    matched_by: str


@dataclass(frozen=True)
class FixtureCrosswalk:
    entries: tuple[FixtureCrosswalkEntry, ...]
    expected: int
    matched: int
    ambiguous: tuple[str, ...]
    unmatched: tuple[str, ...]
    unmapped_team_codes: tuple[int, ...]

    @property
    def ok(self) -> bool:
        return not self.ambiguous and not self.unmatched and not self.unmapped_team_codes

    @property
    def one_to_one(self) -> bool:
        fixtures = [e.official_fixture_id for e in self.entries]
        return len(fixtures) == len(set(fixtures))

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected": int(self.expected),
            "matched": int(self.matched),
            "ambiguous": list(self.ambiguous),
            "unmatched": list(self.unmatched),
            "unmapped_team_codes": list(self.unmapped_team_codes),
            "one_to_one": bool(self.one_to_one),
            "ok": bool(self.ok),
        }


def build_fixture_crosswalk(
    official_fixtures: Sequence[Mapping[str, Any]],
    external_matches: Sequence[Mapping[str, Any]],
    official_teams: Sequence[Mapping[str, Any]],
    *,
    tournament_code: str = PREMIER_LEAGUE_TOURNAMENT_CODE,
) -> FixtureCrosswalk:
    """Map external ``match_id`` onto official fixture ids, one-to-one.

    The external id is NOT an official FPL fixture id.  The deterministic key is
    (season, tournament, home club code, away club code, kickoff) with the
    Gameweek as a corroborating check; every external row must map to exactly one
    official fixture and every official fixture to at most one external row.
    """

    teams_by_code = {
        int(row["code"]): int(row["id"]) for row in official_teams if row.get("code") is not None
    }
    index: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for fixture in official_fixtures:
        key = (
            int(fixture["team_h"]),
            int(fixture["team_a"]),
            normalise_kickoff(fixture.get("kickoff_time")),
        )
        index.setdefault(key, []).append(fixture)

    entries: list[FixtureCrosswalkEntry] = []
    ambiguous: list[str] = []
    unmatched: list[str] = []
    unmapped: set[int] = set()
    for row in external_matches:
        if "tournament" in row and str(row.get("tournament")) != str(tournament_code):
            raise CoreProcessError(
                f"{DIAG_TOURNAMENT_LEAK}: match {row.get('match_id')} is tournament "
                f"{row.get('tournament')!r}, expected {tournament_code!r}",
                reasons=[DIAG_TOURNAMENT_LEAK],
            )
        home_code, away_code = _team_code(row["home_team"]), _team_code(row["away_team"])
        home_id, away_id = teams_by_code.get(home_code), teams_by_code.get(away_code)
        if home_id is None or away_id is None:
            unmapped.update({c for c in (home_code, away_code) if c not in teams_by_code})
            unmatched.append(str(row["match_id"]))
            continue
        key = (home_id, away_id, normalise_kickoff(row.get("kickoff_time")))
        candidates = index.get(key) or []
        if len(candidates) != 1:
            (ambiguous if len(candidates) > 1 else unmatched).append(str(row["match_id"]))
            continue
        fixture = candidates[0]
        event = int(fixture.get("event") or 0)
        if str(row.get("gameweek") or "") not in ("", "None") and int(float(row["gameweek"])) != event:
            ambiguous.append(str(row["match_id"]))
            continue
        entries.append(
            FixtureCrosswalkEntry(
                external_match_id=str(row["match_id"]),
                official_fixture_id=int(fixture["id"]),
                event=event,
                kickoff=normalise_kickoff(row.get("kickoff_time")),
                home_team_id=int(home_id),
                away_team_id=int(away_id),
                matched_by="team_codes+kickoff",
            )
        )
    return FixtureCrosswalk(
        entries=tuple(sorted(entries, key=lambda e: (e.kickoff, e.external_match_id))),
        expected=len(official_fixtures),
        matched=len(entries),
        ambiguous=tuple(sorted(ambiguous)),
        unmatched=tuple(sorted(unmatched)),
        unmapped_team_codes=tuple(sorted(unmapped)),
    )
