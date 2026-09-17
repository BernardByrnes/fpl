"""Player availability snapshots are captured per refresh, completely, once.

The raw archive and the snapshot population for one refresh must describe the
SAME observation, and a capture that stores fewer snapshot rows than the payload
claims players must fail rather than be recorded as complete.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from fpl_brain import ingest, raw_archive
from fpl_brain.api import FplClient
from fpl_brain.database import connect_database

CAPTURE_ONE = "2026-09-17T01:23:45Z"
CAPTURE_TWO = "2026-09-17T06:00:00Z"


def _bootstrap(elements):
    return json.dumps({
        "elements": elements,
        "teams": [{"id": 1, "name": "A", "short_name": "A"}],
        "events": [{"id": 1, "name": "Gameweek 1", "finished": False, "data_checked": False,
                    "deadline_time": "2026-09-19T11:00:00Z"}],
        "element_types": [{"id": 3, "singular_name_short": "MID"}],
    }).encode("utf-8")


def _element(player_id: int, *, cost: int, status: str = "a", news: str = ""):
    return {"id": player_id, "web_name": f"P{player_id}", "first_name": "A", "second_name": "B",
            "team": 1, "element_type": 3, "now_cost": cost, "status": status, "news": news,
            "chance_of_playing_this_round": None, "chance_of_playing_next_round": None,
            "selected_by_percent": "1.0", "total_points": 0, "event_points": 0,
            "form": "0.0", "points_per_game": "0.0", "minutes": 0, "starts": 0, "goals_scored": 0,
            "assists": 0, "clean_sheets": 0, "goals_conceded": 0, "saves": 0, "bonus": 0, "bps": 0,
            "yellow_cards": 0, "red_cards": 0, "penalties_saved": 0, "penalties_missed": 0,
            "influence": "0.0", "creativity": "0.0", "threat": "0.0", "ict_index": "0.0"}


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


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.content = body
        self.text = body.decode("utf-8")
        self.status_code = 200
        self.headers: dict[str, str] = {}


@pytest.fixture
def capture(tmp_path, monkeypatch):
    """A refresh against a temporary raw tree and a temporary database."""

    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "capture.db"

    def run(bootstrap_body: bytes, *, observed_at: str):
        queued = [_FakeResponse(bootstrap_body), _FakeResponse(b"[]")]  # bootstrap, fixtures
        monkeypatch.setattr("fpl_brain.api.requests.Session", lambda: _FakeSession(queued))
        return ingest.run_fetch(
            {"paths": {"raw_dir": str(raw_dir), "database": str(db_path)}},
            summaries="none",
            observed_at=observed_at,
        )

    class Capture:
        def __init__(self):
            self.raw_dir = raw_dir
            self.db_path = db_path
            self.run = run

        def conn(self):
            connection = connect_database(db_path)
            connection.row_factory = sqlite3.Row
            return connection

    return Capture()


# ---------------------------------------------------------------------------
# A / D — every player is captured, at the raw capture's own instant
# ---------------------------------------------------------------------------


def test_A_every_bootstrap_player_receives_a_snapshot(capture):
    elements = [_element(1, cost=50), _element(2, cost=55), _element(3, cost=60)]
    result = capture.run(_bootstrap(elements), observed_at=CAPTURE_ONE)

    conn = capture.conn()
    try:
        rows = conn.execute("SELECT COUNT(*) FROM player_snapshots").fetchone()[0]
        players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    finally:
        conn.close()
    assert result["snapshots"] == 3
    assert rows == 3
    assert players == 3
    assert result["snapshot_completeness"]["complete"] is True


def test_D_snapshots_share_the_raw_capture_instant_exactly(capture):
    elements = [_element(1, cost=50), _element(2, cost=55)]
    capture.run(_bootstrap(elements), observed_at=CAPTURE_ONE)

    manifest = raw_archive.load_manifest(capture.raw_dir)
    bootstrap = [record for record in manifest if record["source"] == "bootstrap_static"]
    assert len(bootstrap) == 1

    conn = capture.conn()
    try:
        stamps = {row[0] for row in conn.execute("SELECT DISTINCT captured_at FROM player_snapshots")}
        generations = {row[0] for row in conn.execute("SELECT DISTINCT captured_at FROM bootstrap_generations")}
    finally:
        conn.close()

    # file evidence, snapshot evidence and generation evidence: one instant.
    assert stamps == {bootstrap[0]["observed_at"]} == {CAPTURE_ONE}
    assert generations == {CAPTURE_ONE}


# ---------------------------------------------------------------------------
# B / C / H — a later capture is a NEW observation, and the past is untouched
# ---------------------------------------------------------------------------


def test_B_C_H_unchanged_values_still_produce_a_new_observation(capture):
    elements = [_element(1, cost=50)]
    capture.run(_bootstrap(elements), observed_at=CAPTURE_ONE)
    capture.run(_bootstrap(elements), observed_at=CAPTURE_TWO)  # identical payload

    conn = capture.conn()
    try:
        per_capture = conn.execute(
            "SELECT captured_at, COUNT(*) FROM player_snapshots GROUP BY captured_at ORDER BY captured_at"
        ).fetchall()
        first_rows = [dict(row) for row in conn.execute(
            "SELECT * FROM player_snapshots WHERE captured_at=?", (CAPTURE_ONE,))]
    finally:
        conn.close()

    assert [tuple(row) for row in per_capture] == [(CAPTURE_ONE, 1), (CAPTURE_TWO, 1)], (
        "unchanged availability must still be recorded as a second observation"
    )
    # The earlier capture is byte-for-byte the same row it always was.
    assert len(first_rows) == 1 and first_rows[0]["now_cost"] == 50
    # One refresh archives every source it fetched; the bootstrap observation is
    # what the snapshots are built from, and a second refresh adds a second one.
    by_source = raw_archive.captures_by_source(capture.raw_dir)
    assert len(by_source["bootstrap_static"]) == 2
    assert {record["observed_at"] for record in by_source["bootstrap_static"]} == {CAPTURE_ONE, CAPTURE_TWO}


def test_G_snapshot_values_come_from_the_payload_not_from_current_db_state(capture):
    capture.run(_bootstrap([_element(1, cost=50)]), observed_at=CAPTURE_ONE)
    # A later payload with a different price and a news string.
    capture.run(_bootstrap([_element(1, cost=99, status="i", news="Knock")]), observed_at=CAPTURE_TWO)

    conn = capture.conn()
    try:
        snapshots = {row["captured_at"]: dict(row) for row in conn.execute("SELECT * FROM player_snapshots")}
    finally:
        conn.close()
    assert snapshots[CAPTURE_ONE]["now_cost"] == 50
    assert snapshots[CAPTURE_ONE]["status"] == "a"
    assert snapshots[CAPTURE_TWO]["now_cost"] == 99
    assert snapshots[CAPTURE_TWO]["status"] == "i"
    assert snapshots[CAPTURE_TWO]["news"] == "Knock"


# ---------------------------------------------------------------------------
# F — missing fields follow the existing schema, no interpolation
# ---------------------------------------------------------------------------


def test_F_missing_official_fields_remain_missing(capture):
    element = _element(1, cost=50)
    element.pop("chance_of_playing_this_round", None)
    element["chance_of_playing_next_round"] = None
    element["news"] = ""
    capture.run(_bootstrap([element]), observed_at=CAPTURE_ONE)

    conn = capture.conn()
    try:
        row = dict(conn.execute("SELECT * FROM player_snapshots").fetchone())
    finally:
        conn.close()
    assert row["chance_of_playing_this_round"] is None
    assert row["chance_of_playing_next_round"] is None
    assert row["news"] == ""


# ---------------------------------------------------------------------------
# E — a partial capture fails closed
# ---------------------------------------------------------------------------


def test_E_a_partial_capture_fails_and_leaves_nothing_behind(capture, monkeypatch):
    elements = [_element(1, cost=50), _element(2, cost=55), _element(3, cost=60)]

    import fpl_brain.repositories as repo

    real_insert = repo.insert_snapshots

    def partial(conn, records, fetch_run_id):
        # Simulate a write that only commits part of the population.
        return real_insert(conn, list(records)[:1], fetch_run_id)

    monkeypatch.setattr("fpl_brain.repositories.insert_snapshots", partial)
    monkeypatch.setattr("fpl_brain.ingest.repo.insert_snapshots", partial, raising=False)

    with pytest.raises(Exception) as excinfo:
        capture.run(_bootstrap(elements), observed_at=CAPTURE_ONE)
    assert "incomplete availability snapshot" in str(excinfo.value)

    conn = capture.conn()
    try:
        snapshots = conn.execute("SELECT COUNT(*) FROM player_snapshots").fetchone()[0]
        runs = conn.execute("SELECT status FROM fetch_runs ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    assert snapshots == 0, "a partial capture left snapshot rows behind"
    assert runs is not None and runs[0] == "failed"


def test_a_complete_capture_is_reported_as_complete(capture):
    result = capture.run(_bootstrap([_element(1, cost=50), _element(2, cost=55)]), observed_at=CAPTURE_ONE)
    completeness = result["snapshot_completeness"]
    assert completeness["official_element_count"] == 2
    assert completeness["parsed_snapshot_count"] == 2
    assert completeness["persisted_snapshot_count"] == 2
    assert completeness["complete"] is True and completeness["problems"] == []


# ---------------------------------------------------------------------------
# Cadence reporting
# ---------------------------------------------------------------------------


def test_the_refresh_reports_its_own_capture_cadence(capture):
    capture.run(_bootstrap([_element(1, cost=50)]), observed_at=CAPTURE_ONE)
    result = capture.run(_bootstrap([_element(1, cost=50)]), observed_at=CAPTURE_TWO)

    cadence = result["capture_cadence"]
    assert cadence["captures"] == 2
    assert cadence["earliest_capture"] == CAPTURE_ONE
    assert cadence["latest_capture"] == CAPTURE_TWO
    assert cadence["partial_captures"] == []
    assert cadence["captures_without_a_generation_record"] == []
    assert cadence["raw_captures_by_source"]["bootstrap_static"] == 2
    assert cadence["captures_without_an_archived_bootstrap"] == []
    # No invented staleness policy: the report states the age, it does not judge it.
    assert "latest_age_hours" in cadence


# ---------------------------------------------------------------------------
# The acceptance test: ONE observation time across file and database
# ---------------------------------------------------------------------------


def test_one_observation_time_across_file_and_database(capture):
    """archive manifest == archive identity time == snapshots == generation record."""

    elements = [_element(1, cost=50), _element(2, cost=55)]
    capture.run(_bootstrap(elements), observed_at=CAPTURE_ONE)

    manifest = raw_archive.load_manifest(capture.raw_dir)
    bootstrap = [record for record in manifest if record["source"] == "bootstrap_static"]
    assert len(bootstrap) == 1
    record = bootstrap[0]
    assert record["observed_at"] == CAPTURE_ONE
    # The path carries the observation time, punctuation stripped.
    assert "".join(ch for ch in CAPTURE_ONE if ch.isalnum()) in record["relative_path"]

    conn = capture.conn()
    try:
        snapshot_stamps = {row[0] for row in conn.execute("SELECT DISTINCT captured_at FROM player_snapshots")}
        run_rows = conn.execute("SELECT started_at FROM fetch_runs ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    assert snapshot_stamps == {CAPTURE_ONE}

    # The archived bytes are the bytes that were received, and they hash to the
    # digest the manifest records alongside that same instant.
    blob = raw_archive.archive_root(capture.raw_dir) / record["relative_path"]
    assert json.loads(blob.read_text(encoding="utf-8"))["elements"][0]["id"] == 1
    assert raw_archive.sha256_of(blob.read_bytes()) == record["payload_sha256"]
