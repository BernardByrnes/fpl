"""Fixture-grain data health: DGW, blank GWs, reschedules, stale placeholders.

The canonical performance grain is `(player_id, event, fixture_id)`.  These
tests prove the grain never loses per-fixture facts, never abandons a blank
gameweek, and cannot be corrupted by schedule placeholders left behind by an
older official payload (for example after a player's club context changed).
"""

from __future__ import annotations

import copy
import json

from fpl_brain import repositories as repo
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    TeamRecord,
)


def _seed_world(conn):
    with conn:
        repo.upsert_teams(
            conn,
            [
                TeamRecord(id=1, name="One"),
                TeamRecord(id=2, name="Two"),
                TeamRecord(id=3, name="Three"),
                TeamRecord(id=10, name="Ten"),
                TeamRecord(id=11, name="Eleven"),
            ],
        )
        repo.upsert_players(conn, [PlayerRecord(id=10, web_name="DGW Player", full_name="DGW Player", team_id=1)])
        repo.upsert_events(
            conn,
            [
                EventRecord(id=1, finished=1, data_checked=1, raw_json={}),
                EventRecord(id=2, finished=1, data_checked=1, raw_json={}),
                EventRecord(id=3, finished=0, data_checked=0, raw_json={}),
            ],
        )
        # Team 1 plays one fixture in GW2, one in GW3, and a DGW pair in GW1.
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, started=1, raw_json={}),
                FixtureRecord(id=102, event=1, team_h=2, team_a=1, finished=1, started=1, raw_json={}),
                FixtureRecord(id=103, event=2, team_h=1, team_a=2, finished=1, started=1, raw_json={}),
                FixtureRecord(id=104, event=3, team_h=1, team_a=2, finished=0, raw_json={}),
            ],
        )


def test_dgw_player_two_fixture_rows_are_both_preserved(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    rows = [
        PlayerGameweekRecord(
            player_id=10, event=1, fixture_id=101, minutes=90, total_points=6,
            expected_goals=0.4, expected_assists=0.1, source="element_summary", raw_json={},
        ),
        PlayerGameweekRecord(
            player_id=10, event=1, fixture_id=102, minutes=75, total_points=3,
            expected_goals=0.2, expected_assists=0.0, source="element_summary", raw_json={},
        ),
    ]
    with conn:
        repo.upsert_player_gameweeks(conn, rows + rows)
    stored = conn.execute(
        "SELECT fixture_id, minutes, total_points FROM player_gameweeks WHERE player_id=10 AND event=1 ORDER BY fixture_id"
    ).fetchall()
    assert [(row[0], row[1], row[2]) for row in stored] == [(101, 90, 6), (102, 75, 3)]
    completed = repo.completed_player_fixture_rows(conn, player_id=10, event=1)
    assert sorted(row["fixture_id"] for row in completed) == [101, 102]
    # Per-fixture xG must remain individually addressable (never merged).
    assert abs(sum(row["expected_goals"] for row in completed) - 0.6) < 1e-9
    conn.close()


def test_bgw_event_stores_no_performance_rows(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    # A blank GW for team 1 has no stored fixture and therefore no rows:
    # the completed-performance boundary stays empty instead of inventing one.
    assert repo.completed_player_fixture_rows(conn, player_id=10, event=99) == []
    sentinel = PlayerGameweekRecord(player_id=10, event=2, fixture_id=-1, minutes=0, total_points=0, source="event_live", raw_json={})
    with conn:
        repo.upsert_player_gameweeks(conn, [sentinel])
        # A real completed summary later replaces the sentinel, never coexists.
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=2, fixture_id=103, minutes=90, total_points=4, source="element_summary", raw_json={})],
        )
    count = conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=2 AND fixture_id=-1"
    ).fetchone()[0]
    assert count == 0
    assert repo.completed_player_fixture_rows(conn, player_id=10, event=2)[0]["fixture_id"] == 103
    conn.close()


def test_future_placeholder_never_downgrades_performance(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=2, fixture_id=103, minutes=90, total_points=4, source="element_summary", raw_json={})],
        )
        # GW3 schedule placeholder (fixture unfinished) must not become an appearance.
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=3, fixture_id=104, minutes=0, source="element_summary", raw_json={})],
        )
    assert repo.completed_player_fixture_rows(conn, player_id=10, event=3) == []
    assert len(repo.completed_player_fixture_rows(conn, player_id=10, event=2)) == 1
    conn.close()


def test_rescheduled_fixture_placeholder_grain_migrates(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        # Old payload claimed fixture 104 in GW3 (pending); officially it is
        # later reassigned to GW4. The new payload claims (4, 104).
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=3, fixture_id=104, minutes=0, source="element_summary", raw_json={})],
        )
        repo.upsert_fixtures(
            conn,
            [FixtureRecord(id=104, event=4, team_h=1, team_a=2, finished=0, raw_json={})],
        )
        repo.prune_stale_element_summary_placeholders(conn, 10, {(4, 104)})
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=4, fixture_id=104, minutes=0, source="element_summary", raw_json={})],
        )
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=3 AND fixture_id=104"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=4 AND fixture_id=104"
    ).fetchone()[0] == 1
    conn.close()


def test_club_change_prunes_unstarted_orphans_and_keeps_owed_ones(tmp_path):
    """Sol F1: the current club is NOT evidence about membership at kickoff.

    Fixture 201 (finished, so it has started) belonged to the player's previous
    club and carries an unresolved placeholder, so it must be RETAINED (fail safe)
    until authoritative history resolves it.  Fixture 202 has not started, so its
    stale placeholder is still pruned, and the performance row survives.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        # Older syncs stored team 2 placeholders from a previous club context.
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=201, event=2, team_h=2, team_a=10, kickoff_time="2026-08-30T13:00:00Z",
                              finished=1, started=1, raw_json={}),
                FixtureRecord(id=202, event=3, team_h=2, team_a=11, finished=0, raw_json={}),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [FixtureRecord(id=102, event=2, team_h=1, team_a=2, kickoff_time="2026-08-30T13:00:00Z",
                           finished=1, started=1, raw_json={})],
        )
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=10, event=2, fixture_id=201, minutes=0, source="element_summary", raw_json={}),
                PlayerGameweekRecord(player_id=10, event=3, fixture_id=202, minutes=0, source="element_summary", raw_json={}),
                # Real performance row for the current club is never pruned even unclaimed.
                PlayerGameweekRecord(player_id=10, event=2, fixture_id=102, minutes=90, total_points=4, source="element_summary", raw_json={}),
            ],
        )
        # No point-in-time capture exists for this player, so membership at the
        # 201 kickoff cannot be disproved -> the owed slot is retained.
        changed = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 103)})
    assert changed == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND fixture_id=201"
    ).fetchone()[0] == 1, "a started placeholder must not be deleted because the club changed"
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND fixture_id=202"
    ).fetchone()[0] == 0
    # The performance row survives and a legitimate current-club placeholder stays.
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND fixture_id=102"
    ).fetchone()[0] == 1
    conn.close()


def test_claimed_placeholder_and_blob_rows_are_never_pruned(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=10, event=3, fixture_id=104, minutes=0, source="element_summary", raw_json={}),
                PlayerGameweekRecord(player_id=10, event=4, fixture_id=-1, minutes=0, total_points=0, source="event_live", raw_json={}),
                PlayerGameweekRecord(player_id=10, event=3, fixture_id=None, minutes=0, total_points=0, source="element_summary", raw_json={}),
            ],
        )
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 104)})
    assert deleted == 0
    conn.close()


# ---------------------------------------------------------------------------
# F1 — a started fixture's placeholder must not vanish without replacement.
#
# A placeholder holds no observation, so its deletion normally only removes a
# stale *slot*.  Once the fixture has started the slot is the sole surviving
# evidence that the player-fixture observation is still owed, and a row that does
# not exist cannot be reported as a gap -- the certified-history audit scans rows,
# not absences.  These three cases pin the replacement-before-delete rule.
# ---------------------------------------------------------------------------


def _placeholder_row(conn, event, fixture_id):
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=event, fixture_id=fixture_id, minutes=0,
                                  source="element_summary", raw_json={})],
        )


def _row_count(conn, event, fixture_id):
    return conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=? AND fixture_id=?",
        (event, fixture_id),
    ).fetchone()[0]


def test_started_current_club_placeholder_without_replacement_is_retained(tmp_path):
    """Case C: fixture 103 (event 2, finished+started) is the player's own club.

    The payload now labels it under event 2 and offers NO replacement row, so the
    stale event-3 slot must survive -- otherwise the completed GW2 observation
    becomes invisible to the audit.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _placeholder_row(conn, 3, 103)
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(2, 103)})
    assert deleted == 0, "a started current-club placeholder must not be pruned without replacement"
    assert _row_count(conn, 3, 103) == 1
    conn.close()


def test_started_current_club_placeholder_with_replacement_is_pruned(tmp_path):
    """Case B: the same stale label may go once the observation is represented."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _placeholder_row(conn, 3, 103)
    _placeholder_row(conn, 2, 103)          # authoritative replacement at the claimed pair
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(2, 103)})
    assert deleted == 1
    assert _row_count(conn, 3, 103) == 0
    assert _row_count(conn, 2, 103) == 1
    conn.close()


def test_unstarted_placeholder_is_pruned_normally(tmp_path):
    """Case A: fixture 104 has not started, so the slot is not yet owed."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _placeholder_row(conn, 3, 104)
    with conn:
        repo.upsert_fixtures(conn, [FixtureRecord(id=104, event=4, team_h=1, team_a=2, finished=0, raw_json={})])
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(4, 104)})
    assert deleted == 1
    assert _row_count(conn, 3, 104) == 0
    conn.close()


# ---------------------------------------------------------------------------
# Sol F1 — membership at kickoff decides, never the player's current club.
#
# Club A (team 1) hosts team 2 in event 2 and the fixture has kicked off.  The
# player now sits at team 11, and an unresolved placeholder for the Club A
# fixture is on file.  Deletion of a STARTED placeholder requires causally valid
# evidence; the current club is not such evidence.
# ---------------------------------------------------------------------------

KICKOFF = "2026-09-06T14:00:00Z"
BEFORE_KICKOFF = "2026-09-05T12:00:00Z"


def _causal_capture(conn, *, club, captured_at, finished_at, accepted=1, player_id=10):
    """One official capture with EXPLICIT causal provenance.

    ``captured_at`` is the request-start stamp ``run_fetch`` writes BEFORE it issues
    the HTTP request; ``finished_at`` is when the response actually completed.  Only
    the latter can authorise anything destructive, and only for an ACCEPTED
    bootstrap generation.
    """

    with conn:
        run_id = conn.execute(
            "INSERT INTO fetch_runs(started_at, finished_at, status, trigger) VALUES (?,?,?,?)",
            (captured_at, finished_at, "success", "fetch_fpl"),
        ).lastrowid
        conn.execute(
            "INSERT INTO bootstrap_generations(fetch_run_id, captured_at, accepted, official_element_count,"
            " parsed_count, persisted_count, element_ids_sha256, element_ids_json, acceptance_rule,"
            " acceptance_rule_version, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, captured_at, int(accepted), 1, 1, 1, "0" * 64, "[]", "test_rule", "v1", finished_at),
        )
        conn.execute(
            "INSERT INTO player_snapshots(player_id, fetch_run_id, captured_at, raw_json) VALUES (?,?,?,?)",
            (player_id, run_id, captured_at, json.dumps({"id": player_id, "team": club})),
        )
    return int(run_id)


def _seed_transfer_world(conn, *, club_at_kickoff=None, club_finished_at=BEFORE_KICKOFF, accepted=1):
    with conn:
        repo.upsert_teams(
            conn,
            [TeamRecord(id=1, name="ClubA"), TeamRecord(id=2, name="Two"), TeamRecord(id=11, name="ClubC")],
        )
        # The player is at Club C NOW; Club A is a former club.
        repo.upsert_players(conn, [PlayerRecord(id=10, web_name="Mover", full_name="Mover", team_id=11)])
        repo.upsert_events(
            conn,
            [
                EventRecord(id=2, finished=1, data_checked=1, raw_json={}),
                EventRecord(id=3, finished=0, data_checked=0, raw_json={}),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [FixtureRecord(id=103, event=2, team_h=1, team_a=2, kickoff_time=KICKOFF,
                           finished=1, started=1, raw_json={})],
        )
    if club_at_kickoff is not None:
        _causal_capture(conn, club=club_at_kickoff, captured_at=BEFORE_KICKOFF,
                        finished_at=club_finished_at, accepted=accepted)


def _placeholder(conn, event, fixture_id):
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=event, fixture_id=fixture_id, minutes=0,
                                  source="element_summary", raw_json={})],
        )


def _rows(conn, fixture_id):
    return conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND fixture_id=?",
        (fixture_id,),
    ).fetchone()[0]


def test_F1A_started_former_club_placeholder_survives_the_transfer(tmp_path):
    """A: joined at kickoff, moved afterwards -> the owed slot must not vanish."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=1)     # official capture before kickoff: Club A
    _placeholder(conn, 2, 103)
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 0
    assert _rows(conn, 103) == 1
    conn.close()


def test_F1B_point_in_time_evidence_of_an_earlier_move_allows_pruning(tmp_path):
    """B: official evidence proves he had already left BEFORE kickoff -> prune."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=11, club_finished_at=BEFORE_KICKOFF)
    _placeholder(conn, 2, 103)
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 1
    assert _rows(conn, 103) == 0
    conn.close()


def test_F1C_started_placeholder_without_club_at_kickoff_proof_is_retained(tmp_path):
    """C: no point-in-time capture -> unprovable -> fail safe (retain)."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=None)
    _placeholder(conn, 2, 103)
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 0
    assert _rows(conn, 103) == 1
    conn.close()


def test_F1D_started_placeholder_with_authoritative_replacement_is_pruned(tmp_path):
    """D: the observation is represented at a claimed pair -> the label may go."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=1)
    _placeholder(conn, 2, 103)
    _placeholder(conn, 3, 103)                        # authoritative replacement row
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 103)})
    assert deleted == 1
    assert _rows(conn, 103) == 1                      # exactly one row left, at the claimed pair
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=2 AND fixture_id=103"
    ).fetchone()[0] == 0
    conn.close()


def test_F1E_not_started_placeholder_prunes_normally(tmp_path):
    """E: an unstarted slot is prospective, not owed -> unchanged behaviour."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _placeholder(conn, 3, 104)
    with conn:
        repo.upsert_fixtures(conn, [FixtureRecord(id=104, event=4, team_h=1, team_a=2, finished=0, raw_json={})])
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(4, 104)})
    assert deleted == 1
    assert _rows(conn, 104) == 0
    conn.close()


# ---------------------------------------------------------------------------
# Sol P2-1 — club-at-kickoff evidence must be CAUSAL.
#
# run_fetch stamps captured_at BEFORE it issues the HTTP request (the accepted
# snapshot shows windows of ~5-8 minutes), so a response received after a fixture
# kicked off can still carry a pre-kickoff captured_at.  Authorising a destructive
# prune therefore requires the response to have COMPLETED by kickoff AND to belong
# to an ACCEPTED bootstrap generation; anything else fails safe.
# ---------------------------------------------------------------------------

AFTER_KICKOFF = "2026-09-06T15:00:00Z"


def test_P2_1A_a_response_completing_after_kickoff_cannot_authorise_deletion(tmp_path):
    """A: request before kickoff, response after -> NOT club-at-kickoff proof."""

    conn = connect_database(tmp_path / "fpl.db")
    # Club C is not in fixture 103 (1 v 2), so this evidence WOULD authorise a prune.
    _seed_transfer_world(conn, club_at_kickoff=11, club_finished_at=AFTER_KICKOFF)
    _placeholder(conn, 2, 103)
    # The leak the old rule allowed: captured_at really is before the kickoff.
    captured_at = conn.execute("SELECT captured_at FROM player_snapshots").fetchone()[0]
    assert captured_at < KICKOFF
    assert repo._official_club_at(conn, 10, KICKOFF) is None
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 0
    assert _rows(conn, 103) == 1
    conn.close()


def test_P2_1B_a_response_completing_before_kickoff_may_authorise_deletion(tmp_path):
    """B: request AND accepted response complete before kickoff -> usable."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=11, club_finished_at=BEFORE_KICKOFF)
    _placeholder(conn, 2, 103)
    assert repo._official_club_at(conn, 10, KICKOFF) == 11
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 1
    assert _rows(conn, 103) == 0
    conn.close()


def test_P2_1C_rejected_generation_evidence_cannot_authorise_deletion(tmp_path):
    """C: a pre-kickoff capture from a REJECTED generation is not authoritative."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=11, club_finished_at=BEFORE_KICKOFF, accepted=0)
    _placeholder(conn, 2, 103)
    assert repo._official_club_at(conn, 10, KICKOFF) is None
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 0
    assert _rows(conn, 103) == 1
    conn.close()


def test_P2_1D_a_later_observation_does_not_retroactively_apply(tmp_path):
    """D: the latest observation AVAILABLE at kickoff decides, not a later one."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=1, club_finished_at=BEFORE_KICKOFF)
    # The transfer is only observed AFTER the fixture kicked off.
    _causal_capture(conn, club=11, captured_at=AFTER_KICKOFF, finished_at=AFTER_KICKOFF)
    _placeholder(conn, 2, 103)
    assert repo._official_club_at(conn, 10, KICKOFF) == 1
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 0, "the later Club C observation must not rewrite kickoff membership"
    assert _rows(conn, 103) == 1
    conn.close()


def test_P2_1E_an_accepted_pre_kickoff_move_may_establish_the_new_club(tmp_path):
    """E: a genuinely earlier, accepted Club C observation is usable evidence."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=None)
    _causal_capture(conn, club=11, captured_at=BEFORE_KICKOFF, finished_at=BEFORE_KICKOFF)
    _placeholder(conn, 2, 103)
    assert repo._official_club_at(conn, 10, KICKOFF) == 11
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 1
    assert _rows(conn, 103) == 0
    conn.close()


def test_P2_1F_without_causal_evidence_the_placeholder_is_retained(tmp_path):
    """F: neither condition proven (rejected generation AND late completion)."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_transfer_world(conn, club_at_kickoff=11, club_finished_at=AFTER_KICKOFF, accepted=0)
    _placeholder(conn, 2, 103)
    assert repo._official_club_at(conn, 10, KICKOFF) is None
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 999)})
    assert deleted == 0
    assert _rows(conn, 103) == 1
    conn.close()


def test_P2_1G_real_ingest_provenance_governs_the_club_evidence(monkeypatch, tmp_path, fixture_json):
    """The REAL ingest path: request-start time is not availability.

    Runs the canonical ``ingest.run_fetch`` (fake HTTP client is the only stub) so
    the fetch_run, the accepted bootstrap generation and the snapshot are written by
    production code, then checks which of the two timestamps the causal lookup may
    trust.  The clock is controlled only to make the leak window deterministic.
    """

    from fpl_brain import ingest

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        def get_bootstrap_static(self):
            return fixture_json("bootstrap_static_sample.json")

        def get_fixtures(self):
            return []

        def close(self):
            return None

    monkeypatch.setattr(ingest, "FplClient", FakeClient)
    monkeypatch.setattr(ingest, "utc_now", lambda: BEFORE_KICKOFF)          # captured_at (pre-request)
    monkeypatch.setattr(repo, "utc_now", lambda: AFTER_KICKOFF)            # started_at / finished_at
    result = ingest.run_fetch(config)
    assert result["status"] == "success", result.get("endpoints_failed")

    conn = connect_database(config["paths"]["database"])
    run = conn.execute("SELECT id, started_at, finished_at FROM fetch_runs ORDER BY id DESC LIMIT 1").fetchone()
    generation = conn.execute(
        "SELECT accepted FROM bootstrap_generations WHERE fetch_run_id=?", (run["id"],)
    ).fetchone()
    snapshot = conn.execute(
        "SELECT player_id, captured_at, raw_json FROM player_snapshots WHERE fetch_run_id=? ORDER BY id LIMIT 1",
        (run["id"],),
    ).fetchone()
    # The real path writes the accepted generation and links it to the run.
    assert generation is not None and generation["accepted"] == 1
    assert snapshot["captured_at"] == BEFORE_KICKOFF < run["finished_at"]
    club = int(json.loads(snapshot["raw_json"])["team"])
    # Request-start time is NOT availability...
    assert repo._official_club_at(conn, snapshot["player_id"], snapshot["captured_at"]) is None
    # ...response completion is.
    assert repo._official_club_at(conn, snapshot["player_id"], run["finished_at"]) == club
    conn.close()
