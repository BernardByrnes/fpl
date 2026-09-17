"""DH-04: capture cadence pairing is bootstrap-authoritative.

Player snapshots derive from ``bootstrap_static``: only an archive record with
``source == "bootstrap_static"`` at the same observed_at pairs with a snapshot
capture.  A fixtures / event-live / element-summary record sharing the instant
proves nothing about the bootstrap side.

The report must also expose the reverse failure
(``bootstrap_without_complete_snapshot``): a bootstrap archive whose database
side holds no accepted COMPLETE generation for the same instant.  Captures
that predate the archive/generation contract are historically unverifiable --
never fabricated as complete, never condemned as fresh gaps, and never
silently labelled healthy because another source happened to share the
instant.

Matrix:
  A. snapshot + bootstrap archive + complete accepted generation -> healthy;
  B. snapshot + fixtures archive + NO bootstrap archive -> forward gap;
  C. bootstrap archive + no snapshots/generation -> reverse gap;
  D. bootstrap archive + partial/rejected generation -> reverse gap;
  E. pre-contract snapshots -> historical, never healthy-by-coincidence.
"""

from __future__ import annotations

import json

from fpl_brain import ingest, raw_archive
from fpl_brain import repositories as repo
from fpl_brain.database import connect_database

CAPTURE_T = "2026-09-17T01:23:45Z"
CAPTURE_NEW = "2026-09-17T06:00:00Z"
CAPTURE_OLD = "2026-01-05T00:00:00Z"


def _bootstrap(elements):
    return json.dumps({
        "elements": elements,
        "teams": [{"id": 1, "name": "A", "short_name": "A"}],
        "events": [{"id": 1, "name": "Gameweek 1", "finished": False, "data_checked": False,
                    "deadline_time": "2026-09-19T11:00:00Z"}],
        "element_types": [{"id": 3, "singular_name_short": "MID"}],
    }).encode("utf-8")


def _element(player_id: int, *, cost: int):
    return {"id": player_id, "web_name": f"P{player_id}", "first_name": "A", "second_name": "B",
            "team": 1, "element_type": 3, "now_cost": cost, "status": "a", "news": "",
            "chance_of_playing_this_round": None, "chance_of_playing_next_round": None,
            "selected_by_percent": "1.0", "total_points": 0, "event_points": 0,
            "form": "0.0", "points_per_game": "0.0", "minutes": 0, "starts": 0, "goals_scored": 0,
            "assists": 0, "clean_sheets": 0, "goals_conceded": 0, "saves": 0, "bonus": 0, "bps": 0,
            "yellow_cards": 0, "red_cards": 0, "penalties_saved": 0, "penalties_missed": 0,
            "influence": "0.0", "creativity": "0.0", "threat": "0.0", "ict_index": "0.0"}


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.content = body
        self.text = body.decode("utf-8")
        self.status_code = 200
        self.headers: dict[str, str] = {}


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.headers: dict[str, str] = {}

    def get(self, url, timeout=None):
        if not self._responses:
            raise AssertionError("no queued response for " + url)
        return self._responses.pop(0)

    def close(self):
        pass


def _fetch(raw_dir, db_path, bootstrap_body, *, observed_at, monkeypatch):
    queued = [_FakeResponse(bootstrap_body), _FakeResponse(b"[]")]  # bootstrap, fixtures
    monkeypatch.setattr("fpl_brain.api.requests.Session", lambda: _FakeSession(queued))
    return ingest.run_fetch(
        {"paths": {"raw_dir": str(raw_dir), "database": str(db_path)}},
        summaries="none",
        observed_at=observed_at,
    )


def _health(db_path, raw_dir):
    conn = connect_database(db_path)
    try:
        return raw_archive.capture_cadence_health(conn, raw_dir=raw_dir)
    finally:
        conn.close()


def _insert_legacy_snapshots(db_path, *, captured_at, count):
    """Snapshots with no generation record: the pre-contract historical shape.

    Foreign keys are relaxed for this setup-only insert because the legacy row
    deliberately has no players/fetch-run ancestry; the report reads snapshot
    state, never FK integrity, and production code is untouched.
    """

    conn = connect_database(db_path)
    try:
        # No pending transaction here, so the pragma takes effect; it is
        # restored before returning.
        conn.execute("PRAGMA foreign_keys=OFF")
        with conn:
            run_id = repo.create_fetch_run(conn, "fetch_fpl", None)
            for player_id in range(1, count + 1):
                conn.execute(
                    "INSERT INTO player_snapshots(player_id, fetch_run_id, captured_at, raw_json)"
                    " VALUES (?,?,?,?)",
                    (player_id, run_id, captured_at, "{}"),
                )
        conn.execute("PRAGMA foreign_keys=ON")
    finally:
        conn.close()


def _record_generation(db_path, *, captured_at, accepted, official, parsed, persisted):
    conn = connect_database(db_path)
    try:
        with conn:
            repo.record_bootstrap_generation(
                conn,
                captured_at=captured_at,
                accepted=accepted,
                official_element_count=official,
                parsed_count=parsed,
                persisted_count=persisted,
                element_ids=list(range(1, official + 1)),
                element_ids_sha256="test-sha",
                acceptance_rule="FIRST_GENERATION" if accepted else "REJECTED",
                acceptance_rule_version="BOOTSTRAP_GENERATION_ACCEPTANCE v1",
                rejection_reasons=[] if accepted else ["R1_EMPTY_PAYLOAD"],
            )
    finally:
        conn.close()


def test_DH04_A_healthy_pair_is_not_a_gap(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "capture.db"
    _fetch(raw_dir, db_path, _bootstrap([_element(1, cost=50)]), observed_at=CAPTURE_T, monkeypatch=monkeypatch)

    health = _health(db_path, raw_dir)
    assert health["snapshot_without_bootstrap"] == []
    assert health["captures_without_an_archived_bootstrap"] == []
    assert health["bootstrap_without_complete_snapshot"] == []
    assert health["historically_unverifiable_captures"] == []


def test_DH04_B_fixtures_archive_does_not_pair_snapshots(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "capture.db"
    _fetch(raw_dir, db_path, _bootstrap([_element(1, cost=50)]), observed_at=CAPTURE_T, monkeypatch=monkeypatch)

    # A fixtures-only view of the same instant: the bootstrap side is absent.
    fixtures_only = tmp_path / "fixtures_only"
    raw_archive.archive_raw_capture(
        fixtures_only, source="fixtures", observed_at=CAPTURE_T, body=b"[]")

    health = _health(db_path, fixtures_only)
    assert health["snapshot_without_bootstrap"] == [CAPTURE_T]
    assert health["captures_without_an_archived_bootstrap"] == [CAPTURE_T]
    assert health["historically_unverifiable_captures"] == []


def test_DH04_C_bootstrap_without_any_capture_is_a_reverse_gap(tmp_path):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "capture.db"
    connect_database(db_path).close()
    raw_archive.archive_raw_capture(
        raw_dir, source="bootstrap_static", observed_at=CAPTURE_T,
        body=_bootstrap([_element(1, cost=50)]))

    health = _health(db_path, raw_dir)
    assert health["bootstrap_without_complete_snapshot"] == [CAPTURE_T]
    assert health["snapshot_without_bootstrap"] == []


def test_DH04_D_rejected_generation_is_a_reverse_gap(tmp_path):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "capture.db"
    connect_database(db_path).close()
    raw_archive.archive_raw_capture(
        raw_dir, source="bootstrap_static", observed_at=CAPTURE_T,
        body=_bootstrap([_element(1, cost=50), _element(2, cost=55)]))
    _record_generation(db_path, captured_at=CAPTURE_T, accepted=False,
                       official=2, parsed=2, persisted=2)

    health = _health(db_path, raw_dir)
    assert health["bootstrap_without_complete_snapshot"] == [CAPTURE_T]


def test_DH04_D_partial_generation_is_a_reverse_gap(tmp_path):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "capture.db"
    connect_database(db_path).close()
    raw_archive.archive_raw_capture(
        raw_dir, source="bootstrap_static", observed_at=CAPTURE_T,
        body=_bootstrap([_element(1, cost=50), _element(2, cost=55), _element(3, cost=60)]))
    _record_generation(db_path, captured_at=CAPTURE_T, accepted=True,
                       official=3, parsed=3, persisted=1)

    health = _health(db_path, raw_dir)
    assert health["bootstrap_without_complete_snapshot"] == [CAPTURE_T]


def test_DH04_E_pre_contract_snapshots_are_historical_not_gaps(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "capture.db"
    connect_database(db_path).close()
    _insert_legacy_snapshots(db_path, captured_at=CAPTURE_OLD, count=2)

    health = _health(db_path, raw_dir)
    assert health["historically_unverifiable_captures"] == [CAPTURE_OLD]
    assert health["snapshot_without_bootstrap"] == [], "old data must not be condemned as a fresh gap"
    assert health["bootstrap_without_complete_snapshot"] == []


def test_DH04_E_coincidental_source_is_not_a_healthy_pair(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "capture.db"
    # New-era healthy capture establishes the archive/generation era ...
    _fetch(raw_dir, db_path, _bootstrap([_element(1, cost=50)]), observed_at=CAPTURE_NEW, monkeypatch=monkeypatch)
    # ... while an old snapshot shares its instant with only a fixtures record.
    _insert_legacy_snapshots(db_path, captured_at=CAPTURE_OLD, count=2)
    raw_archive.archive_raw_capture(
        raw_dir, source="fixtures", observed_at=CAPTURE_OLD, body=b"[]")

    health = _health(db_path, raw_dir)
    assert CAPTURE_OLD in health["historically_unverifiable_captures"]
    assert CAPTURE_OLD not in health["snapshot_without_bootstrap"]
    assert health["snapshot_without_bootstrap"] == [], "the new-era capture is paired; the old one is historical"
    assert health["bootstrap_without_complete_snapshot"] == []
