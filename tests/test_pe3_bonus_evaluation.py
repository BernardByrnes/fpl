"""PE-3 — structural bonus evaluation contracts (behavioural, fail-closed).

Every test builds its OWN temporary database and raw-bootstrap tree, so the contracts are
exercised without depending on the live workspace.  The point of this suite is the
FAIL-CLOSED behaviour: a wrong authority, a mutated raw file, an incoherent chain or a
missing outcome must raise or be excluded, never be silently accepted or read as zero.

FIXTURE FIDELITY.  The database is created through ``database.connect_database`` so the
tests run against the CURRENT canonical migrated schema, and the rows are built from
``repositories.GAMEWEEK_PERFORMANCE_COLUMNS``, so a test fixture cannot drift away from
production column names or from the canonical placeholder predicate.  The synthetic legacy
bootstraps are pushed through the SAME ``parsers.parse_bootstrap`` /
``parsers.validate_bootstrap_payload`` path the producer uses, and each snapshot's
``raw_json`` IS the serialisation of the element object itself rather than a hand-built
lookalike.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import bonus_allocation as ba
from fpl_brain import bonus_challenger as bc
from fpl_brain import bonus_evaluation as be
from fpl_brain import parsers, repositories
from fpl_brain.database import connect_database

FETCH_ID = 7
PREVIOUS_FETCH_ID = 6
TARGET_EVENT = 4
CUTOFF = "2026-09-12T10:40:04Z"
DEADLINE = "2026-09-12T12:30:00Z"
FINISHED = "2026-09-11T22:45:41Z"
CAPTURED = "2026-09-11T22:45:40Z"
PREVIOUS_FINISHED = "2026-09-11T20:00:01Z"
PREVIOUS_CAPTURED = "2026-09-11T20:00:00Z"
KICKOFF = "2026-09-12T14:00:00Z"

#: The four frozen predictive runs, as declared for the GW4 replay.
CHAIN = {"xpts": 133, "minutes": 127, "team": 128, "rate": 130}
_CHAIN_FAMILIES = {"xpts": "xpts_v1", "minutes": "minutes_v1",
                   "team": "team_strength_v1", "rate": "player_rates_v1"}

_FLOAT_PERFORMANCE_COLUMNS = frozenset({
    "influence", "creativity", "threat", "ict_index", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded"})


# ---------------------------------------------------------------------------
# Canonical bootstrap factory — the fixture must satisfy the PRODUCTION validator
# ---------------------------------------------------------------------------


def _element(pid: int, team: int, element_type: int, minutes: int, bps: int, *,
             bonus: int = 0) -> dict:
    return {"id": int(pid), "team": int(team), "element_type": int(element_type),
            "minutes": int(minutes), "bps": int(bps), "bonus": int(bonus),
            "now_cost": 50, "status": "a", "web_name": f"P{pid}"}


def _bootstrap_payload(elements) -> dict:
    """A synthetic bootstrap shaped like a real one, with teams derived from the elements."""

    team_ids = sorted({int(item["team"]) for item in elements} | {1, 2})
    return {"elements": [dict(item) for item in elements],
            "teams": [{"id": t, "name": f"Team {t}", "short_name": f"T{t}"} for t in team_ids],
            "events": [{"id": e, "name": f"GW{e}"} for e in range(1, 39)],
            "element_types": [{"id": i, "singular_name": name} for i, name in
                              ((1, "Goalkeeper"), (2, "Defender"), (3, "Midfielder"),
                               (4, "Forward"))]}


def _canonical_records(payload: dict, *, captured_at: str, event_context: int):
    """Push the synthetic payload through the PRODUCTION parse + validation path.

    Canonical validation is NOT mocked away: a fixture that cannot survive the parser the
    producer runs is not evidence about the producer.
    """

    records = parsers.parse_bootstrap(payload, captured_at, event_context=event_context)
    parsers.validate_bootstrap_payload(payload, records)
    assert len(records.players) == len(payload["elements"]), "fixture parses incompletely"
    assert len(records.snapshots) == len(payload["elements"]), "fixture snapshots incomplete"
    return records


def _write_raw(raw_dir: Path, fetch_id: int, elements, *, finished_at: str,
               event_context: int = 3) -> Path:
    path = raw_dir / str(fetch_id)
    path.mkdir(parents=True, exist_ok=True)
    payload = _bootstrap_payload(elements)
    _canonical_records(payload, captured_at=finished_at, event_context=event_context)
    target = path / "bootstrap_static.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Canonical player_gameweek row helpers
# ---------------------------------------------------------------------------


def performance_zeroes() -> dict:
    """An explicit ZERO for EVERY canonical performance column.

    A genuine completed did-not-play carries this shape (``minutes=0``, ``starts=0``,
    ``bonus=0``, ``bps=0`` and an explicit zero everywhere else), which is what makes it a
    valid realised zero rather than a schedule placeholder.
    """

    return {name: (0.0 if name in _FLOAT_PERFORMANCE_COLUMNS else 0)
            for name in repositories.GAMEWEEK_PERFORMANCE_COLUMNS}


def placeholder_row() -> dict:
    """``minutes = 0`` and EVERY other canonical performance column NULL."""

    row = {name: None for name in repositories.GAMEWEEK_PERFORMANCE_COLUMNS}
    row["minutes"] = 0
    return row


def _insert_gameweek(conn: sqlite3.Connection, player_id: int, fixture_id: int,
                     performance: dict, *, source: str = "element_summary_detail") -> None:
    row = {"player_id": int(player_id), "event": TARGET_EVENT, "fixture_id": int(fixture_id),
           "opponent_team": 2, "was_home": 1, "kickoff_time": KICKOFF,
           "value": 50, "selected": 0, "transfers_in": 0, "transfers_out": 0,
           "transfers_balance": 0, "source": source, "raw_json": "{}",
           "updated_at": "2026-09-12T15:00:00Z"}
    row.update(performance)
    placeholders = ", ".join("?" * len(row))
    conn.execute(f"INSERT INTO player_gameweeks ({', '.join(row)}) VALUES ({placeholders})",
                 list(row.values()))
    conn.commit()


# ---------------------------------------------------------------------------
# The synthetic database + raw tree
# ---------------------------------------------------------------------------


def _default_elements() -> list[dict]:
    return [_element(1, 1, 1, 900, 300), _element(2, 1, 2, 800, 200),
            _element(3, 2, 3, 700, 150)]


def _club_population(*, clubs: int, per_club: int, exclude_clubs=()) -> list[dict]:
    """A population whose CLUB MEMBERSHIP is encoded in the raw element objects."""

    excluded = {int(c) for c in exclude_clubs}
    elements, pid = [], 0
    for club in range(1, clubs + 1):
        for _ in range(per_club):
            pid += 1
            elements.append(_element(pid, club, 2, 800, 100))
    return [e for e in elements if int(e["team"]) not in excluded]


def _seed_reference(conn: sqlite3.Connection, *populations) -> None:
    """Seed the durable reference tables from every population, OLDEST first.

    ``players`` is the durable identity table, so a player who has dropped out of the
    CURRENT bootstrap still has a row — which is what lets the previous fetch's snapshots
    satisfy their foreign key.  Passing the current population LAST makes it win on
    conflict, so the current table reflects current state.
    """

    team_ids = {1, 2}
    for population in populations:
        team_ids |= {int(item["team"]) for item in population}
    for team_id in sorted(team_ids):
        conn.execute(
            "INSERT OR REPLACE INTO teams (id, name, short_name, raw_json, updated_at)"
            " VALUES (?,?,?,?,?)", (team_id, f"Team {team_id}", f"T{team_id}", "{}", FINISHED))
    for position_id, name in ((1, "Goalkeeper"), (2, "Defender"), (3, "Midfielder"),
                              (4, "Forward")):
        conn.execute(
            "INSERT OR REPLACE INTO positions (id, singular_name, raw_json, updated_at)"
            " VALUES (?,?,?,?)", (position_id, name, "{}", FINISHED))
    for population in reversed(populations):
        for item in population:
            conn.execute(
                "INSERT OR REPLACE INTO players (id, web_name, team_id, element_type,"
                " first_seen_at, last_seen_at, is_active, raw_json, updated_at)"
                " VALUES (?,?,?,?,?,?,1,?,?)",
                (int(item["id"]), str(item["web_name"]), int(item["team"]),
                 int(item["element_type"]), FINISHED, FINISHED, json.dumps(item), FINISHED))


def _insert_fetch_run(conn: sqlite3.Connection, fetch_id: int, started_at: str,
                      finished_at: str, status: str, raw_dir: Path) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO fetch_runs (id, started_at, finished_at, status, trigger,"
        " current_event, endpoints_ok, endpoints_failed, error_message, raw_dir)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (int(fetch_id), started_at, finished_at, status, "fetch_fpl", 3,
         '["bootstrap-static"]', "[]", None, str(raw_dir)))


def _insert_snapshots(conn: sqlite3.Connection, fetch_id: int, elements, captured_at: str,
                      *, event_context: int = 3) -> None:
    """``raw_json`` IS the serialisation of the matching element, never a lookalike."""

    for item in elements:
        conn.execute(
            "INSERT INTO player_snapshots (player_id, fetch_run_id, captured_at,"
            " event_context, minutes, bps, bonus, status, raw_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (int(item["id"]), int(fetch_id), captured_at, int(event_context),
             int(item["minutes"]), int(item["bps"]), int(item["bonus"]),
             str(item["status"]), json.dumps(item)))


def _insert_target_fixtures(conn: sqlite3.Connection, kickoffs, *, event=TARGET_EVENT,
                            finished: int = 1, started: int = 1, teams=(1, 2)) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO events (id, name, finished, data_checked, raw_json, updated_at)"
        " VALUES (?,?,1,1,'{}',?)", (int(event), f"GW{event}", FINISHED))
    for index, kickoff in enumerate(kickoffs, start=100):
        conn.execute(
            "INSERT OR REPLACE INTO fixtures (id, event, kickoff_time, team_h, team_a,"
            " started, finished, raw_json, updated_at) VALUES (?,?,?,?,?,?,?,'{}',?)",
            (index, int(event), kickoff, int(teams[0]), int(teams[1]), int(started),
             int(finished), FINISHED))


_BUILD_SEQUENCE = itertools.count(1)


def _build(tmp_path: Path, *, elements=None, previous_elements=None, snapshots=None,
           previous_snapshots=None, fetch_status="success", previous_status="success",
           finished=FINISHED, captured=CAPTURED, previous_finished=PREVIOUS_FINISHED,
           previous_captured=PREVIOUS_CAPTURED, kickoffs=(KICKOFF,),
           current_teams=None):
    """A temporary DB + raw tree whose fetch 7 (and optional fetch 6) satisfy every gate.

    Each call gets its OWN database file, so a test that needs two independent
    populations (before/after) does not collide with itself.
    """

    elements = [dict(e) for e in (elements if elements is not None else _default_elements())]
    previous = None if previous_elements is None else [dict(e) for e in previous_elements]
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, FETCH_ID, elements, finished_at=finished)
    if previous is not None:
        _write_raw(raw_dir, PREVIOUS_FETCH_ID, previous, finished_at=previous_finished)

    conn = connect_database(tmp_path / f"test_{next(_BUILD_SEQUENCE)}.db")
    _seed_reference(conn, elements, previous or [])
    _insert_fetch_run(conn, FETCH_ID, CAPTURED, finished, fetch_status, raw_dir)
    if previous is not None:
        _insert_fetch_run(conn, PREVIOUS_FETCH_ID, PREVIOUS_CAPTURED, previous_finished,
                          previous_status, raw_dir)
    _insert_snapshots(conn, FETCH_ID, snapshots if snapshots is not None else elements, captured)
    if previous is not None:
        _insert_snapshots(conn, PREVIOUS_FETCH_ID,
                          previous_snapshots if previous_snapshots is not None else previous,
                          previous_captured)
    _insert_target_fixtures(conn, kickoffs)
    if current_teams:
        for player_id, team_id in current_teams.items():
            conn.execute("UPDATE players SET team_id = ? WHERE id = ?",
                         (int(team_id), int(player_id)))
    conn.commit()
    return conn, raw_dir


def _verify(conn, raw_dir, **over):
    kwargs = dict(fetch_run_id=FETCH_ID, raw_dir=raw_dir, deadline=DEADLINE,
                  target_event=TARGET_EVENT)
    kwargs.update(over)
    return be.verify_legacy_bootstrap_generation(conn, **kwargs)


# ---------------------------------------------------------------------------
# LEGACY AUTHORITY
# ---------------------------------------------------------------------------


def test_successful_reconstructed_legacy_authority(tmp_path):
    conn, raw_dir = _build(tmp_path)
    evidence = _verify(conn, raw_dir)
    assert evidence.authority_mode == be.LEGACY_RECONSTRUCTED_COMPLETE
    assert evidence.formal_generation_record == "NONE"
    assert evidence.as_dict()["generation_accepted"] is None, "never claim accepted=true"
    assert evidence.raw_elements == evidence.snapshots == evidence.raw_json_compared == 3
    assert evidence.raw_json_matches == 3
    assert evidence.raw_id_digest == evidence.snapshot_id_digest


def test_first_legacy_fetch_with_no_predecessor_is_accepted(tmp_path):
    """A fetch with no PROVEN predecessor has no population baseline, and must not crash.

    The first fetch of a season legitimately has nothing before it: the population gates
    are then declared inapplicable rather than evaluated against an empty baseline.
    """

    conn, raw_dir = _build(tmp_path)
    evidence = _verify(conn, raw_dir)
    assert evidence.previous_fetch_run_id is None
    assert evidence.previous_population is None
    assert evidence.retained_fraction is None
    assert evidence.absolute_drop is None
    assert evidence.whole_club_loss is None


def test_raw_file_missing_fails(tmp_path):
    conn, raw_dir = _build(tmp_path)
    (raw_dir / str(FETCH_ID) / "bootstrap_static.json").unlink()
    with pytest.raises(be.BackgroundAuthorityError, match="raw bootstrap missing"):
        _verify(conn, raw_dir)


def test_sha_mutation_fails(tmp_path):
    conn, raw_dir = _build(tmp_path)
    with pytest.raises(be.BackgroundAuthorityError, match="does not match the frozen value"):
        _verify(conn, raw_dir, expected_raw_sha256="0" * 64)


def test_malformed_bootstrap_fails(tmp_path):
    conn, raw_dir = _build(tmp_path)
    (raw_dir / str(FETCH_ID) / "bootstrap_static.json").write_text(
        json.dumps({"elements": [], "teams": []}), encoding="utf-8")
    with pytest.raises(be.BackgroundAuthorityError, match="canonical bootstrap validation"):
        _verify(conn, raw_dir)


def test_element_without_web_name_fails_canonical_validation(tmp_path):
    """The canonical validator requires a name on every element; the gate must surface it."""

    conn, raw_dir = _build(tmp_path)
    payload = json.loads((raw_dir / str(FETCH_ID) / "bootstrap_static.json").read_text())
    del payload["elements"][0]["web_name"]
    (raw_dir / str(FETCH_ID) / "bootstrap_static.json").write_text(json.dumps(payload),
                                                                   encoding="utf-8")
    with pytest.raises(be.BackgroundAuthorityError, match="canonical bootstrap validation"):
        _verify(conn, raw_dir)


def test_raw_count_mismatch_fails(tmp_path):
    conn, raw_dir = _build(tmp_path)
    conn.execute("DELETE FROM player_snapshots WHERE player_id = 3")
    conn.commit()
    with pytest.raises(be.BackgroundAuthorityError, match="!= snapshot count"):
        _verify(conn, raw_dir)


def test_raw_id_set_mismatch_fails(tmp_path):
    """Equal COUNTS with a different ID set must still fail: identity, not cardinality."""

    conn, raw_dir = _build(tmp_path)
    conn.execute(
        "INSERT INTO players (id, web_name, team_id, element_type, first_seen_at,"
        " last_seen_at, is_active, raw_json, updated_at) VALUES (4,'P4',1,2,?,?,1,'{}',?)",
        (FINISHED, FINISHED, FINISHED))
    conn.execute("DELETE FROM player_snapshots WHERE player_id = 3")
    conn.execute(
        "INSERT INTO player_snapshots (player_id, fetch_run_id, captured_at, event_context,"
        " minutes, bps, bonus, status, raw_json)"
        " SELECT 4, fetch_run_id, captured_at, event_context, minutes, bps, bonus, status,"
        " raw_json FROM player_snapshots WHERE player_id = 1")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM player_snapshots WHERE fetch_run_id = ?",
                        (FETCH_ID,)).fetchone()[0] == 3, "counts must still agree"
    with pytest.raises(be.BackgroundAuthorityError, match="ID set"):
        _verify(conn, raw_dir)


def test_raw_json_mismatch_fails(tmp_path):
    conn, raw_dir = _build(tmp_path)
    conn.execute("UPDATE player_snapshots SET raw_json = ? WHERE player_id = 2",
                 (json.dumps(_element(2, 1, 2, 800, 999)),))
    conn.commit()
    with pytest.raises(be.BackgroundAuthorityError, match="differ from their raw bootstrap"):
        _verify(conn, raw_dir)


def test_multiple_capture_instants_fail(tmp_path):
    conn, raw_dir = _build(tmp_path)
    conn.execute("UPDATE player_snapshots SET captured_at = '2026-09-11T23:00:00Z'"
                 " WHERE player_id = 1")
    conn.commit()
    with pytest.raises(be.BackgroundAuthorityError, match="one coherent instant"):
        _verify(conn, raw_dir)


def test_fetch_status_not_success_fails(tmp_path):
    conn, raw_dir = _build(tmp_path, fetch_status="failed")
    with pytest.raises(be.BackgroundAuthorityError, match="not success"):
        _verify(conn, raw_dir)


def test_fetch_after_deadline_fails(tmp_path):
    conn, raw_dir = _build(tmp_path, finished="2026-09-12T13:00:00Z")
    with pytest.raises(be.BackgroundAuthorityError, match="not before the deadline"):
        _verify(conn, raw_dir)


def test_missing_target_kickoff_fails(tmp_path):
    conn, raw_dir = _build(tmp_path, kickoffs=(None,))
    with pytest.raises(be.BackgroundAuthorityError, match="missing/unparseable kickoff"):
        _verify(conn, raw_dir)


def test_unparseable_target_kickoff_fails(tmp_path):
    conn, raw_dir = _build(tmp_path, kickoffs=("not-a-time",))
    with pytest.raises(be.BackgroundAuthorityError, match="missing/unparseable kickoff"):
        _verify(conn, raw_dir)


def test_fixture_started_before_fetch_fails(tmp_path):
    conn, raw_dir = _build(tmp_path, kickoffs=("2026-09-11T22:00:00Z",))
    with pytest.raises(be.BackgroundAuthorityError, match="kicked off before the fetch"):
        _verify(conn, raw_dir)


def test_no_target_fixtures_fails(tmp_path):
    conn, raw_dir = _build(tmp_path, kickoffs=())
    with pytest.raises(be.BackgroundAuthorityError, match="no fixture rows"):
        _verify(conn, raw_dir)


def test_retained_fraction_below_threshold_fails(tmp_path):
    """§8: the diagnostic must GATE, not merely be reported.

    Both payloads are independently canonical and exactly match their OWN snapshots, so the
    only usable rejection is the retention gate (100 -> 50 players keeps the absolute drop
    at 50, below the 75 ceiling).
    """

    previous = [_element(i, 1, 2, 800, 100) for i in range(1, 101)]
    current = previous[:50]
    conn, raw_dir = _build(tmp_path, elements=current, previous_elements=previous)
    with pytest.raises(be.BackgroundAuthorityError, match="retained fraction"):
        _verify(conn, raw_dir)


def test_absolute_drop_above_threshold_fails(tmp_path):
    """§11: the ABSOLUTE-drop gate, isolated from the retention and club-loss gates.

    56 clubs x 50 players = 2800.  Removing the first two players of each of the first 40
    clubs removes 80 (> 75) while every club keeps 48 members, and
    retained = 2720/2800 = 0.9714 >= 0.97.  So only the absolute gate can fire.
    """

    previous = _club_population(clubs=56, per_club=50)
    dropped = {(club - 1) * 50 + offset for club in range(1, 41) for offset in (1, 2)}
    current = [e for e in previous if int(e["id"]) not in dropped]
    assert len(dropped) == 80
    assert len({int(e["team"]) for e in current}) == 56, "no club was emptied"
    conn, raw_dir = _build(tmp_path, elements=current, previous_elements=previous)
    with pytest.raises(be.BackgroundAuthorityError, match="absolute drop"):
        _verify(conn, raw_dir)


def test_historical_whole_club_loss_fails(tmp_path):
    """§9: club membership is encoded in the HISTORICAL raw element objects.

    40 clubs x 5 players = 200.  Dropping one entire club removes 5 players, so
    retained = 195/200 = 0.975 >= 0.97 and the absolute drop is 5 <= 75: both population
    thresholds PASS, leaving the whole-club gate as the only one that can fire.
    """

    previous = _club_population(clubs=40, per_club=5)
    current = _club_population(clubs=40, per_club=5, exclude_clubs=(1,))
    conn, raw_dir = _build(tmp_path, elements=current, previous_elements=previous)
    with pytest.raises(be.BackgroundAuthorityError, match="whole club disappeared"):
        _verify(conn, raw_dir)


def test_partial_player_removal_is_not_a_whole_club_loss():
    """§3 PREDECESSOR KILL (direct): a club that still has members has not disappeared.

    On b4e8f996 the implementation intersected the removed player-id set with the surviving
    player-id set — two sets that are disjoint BY CONSTRUCTION, so the intersection was always
    empty and ANY single removal was reported as a lost club.  The discriminating case is the
    last assertion, which is the call shape the old verifier actually made.
    """

    previous = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}
    current = {pid: team for pid, team in previous.items() if pid != 1}
    assert be._whole_club_loss(previous, current) is False
    assert be._whole_club_loss(previous, previous) is False, "no change is never a loss"
    # Club 1 lost exactly one member and keeps four, so it did not vanish.  Written with the
    # REMOVED player as the left map, this is the case b4e8f996 got wrong.
    assert be._whole_club_loss({1: 1}, current) is False


def test_a_vanished_club_identity_is_a_whole_club_loss():
    previous = {1: 1, 2: 1, 3: 2, 4: 2}
    current = {3: 2, 4: 2}                      # club 1 has no surviving member
    assert be._whole_club_loss(previous, current) is True


def test_a_new_club_is_not_a_whole_club_loss():
    """The test is one-directional: a club APPEARING is not a club disappearing."""

    previous = {1: 1, 2: 1}
    current = {1: 1, 2: 1, 3: 2}
    assert be._whole_club_loss(previous, current) is False


def test_verifier_accepts_a_population_that_lost_exactly_one_player(tmp_path):
    """§3 at the VERIFIER level: a one-player drop must not be rejected as a club loss.

    The fixture is sized so the retention gate passes (199/200 = 0.995) and the absolute drop
    is 1, leaving whole-club loss as the only gate that could reject it.
    """

    previous = _club_population(clubs=40, per_club=5)
    current = [e for e in previous if int(e["id"]) != 1]
    conn, raw_dir = _build(tmp_path, elements=current, previous_elements=previous)
    evidence = _verify(conn, raw_dir)
    assert evidence.whole_club_loss is False
    assert evidence.retained_fraction == pytest.approx(199 / 200)
    assert evidence.absolute_drop == 1
    assert evidence.previous_population == 200


def test_current_players_team_id_cannot_alter_historical_club_loss(tmp_path):
    """The CURRENT players table must not decide an old population's club loss.

    The current table is mutated to claim every player now belongs to one club, which is
    exactly the state that would make a CURRENT-STATE check see survivors for the vanished
    club.  The historical result must be unchanged.
    """

    previous = _club_population(clubs=40, per_club=5)
    current = _club_population(clubs=40, per_club=5, exclude_clubs=(1,))
    merged = {int(e["id"]): 2 for e in previous}
    conn, raw_dir = _build(tmp_path, elements=current, previous_elements=previous,
                           current_teams=merged)
    assert {int(r["team_id"]) for r in conn.execute("SELECT team_id FROM players")} == {2}
    with pytest.raises(be.BackgroundAuthorityError, match="whole club disappeared"):
        _verify(conn, raw_dir)


def test_unproven_previous_fetch_is_skipped_not_trusted(tmp_path):
    """A predecessor whose snapshots disagree with its raw payload cannot set the baseline."""

    previous = _club_population(clubs=40, per_club=5)
    current = _club_population(clubs=40, per_club=5, exclude_clubs=(1,))
    # The previous fetch's SNAPSHOTS lose a player, so its population is unproven.
    conn, raw_dir = _build(tmp_path, elements=current, previous_elements=previous,
                           previous_snapshots=previous[:-1])
    evidence = _verify(conn, raw_dir)
    assert evidence.previous_fetch_run_id is None, "an unproven predecessor is not authority"
    assert evidence.whole_club_loss is None


# ---------------------------------------------------------------------------
# BACKGROUND
# ---------------------------------------------------------------------------


def _backgrounds(tmp_path, elements):
    conn, _raw = _build(tmp_path, elements=elements)
    return be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)[0]


def test_positive_minutes_with_null_bps_fails(tmp_path):
    conn, _raw = _build(tmp_path)
    conn.execute("UPDATE player_snapshots SET bps = NULL WHERE player_id = 1")
    conn.commit()
    with pytest.raises(be.BackgroundEvidenceError, match="no BPS"):
        be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)


def test_positive_minutes_with_zero_bps_is_valid(tmp_path):
    evidence = _backgrounds(tmp_path, [_element(1, 1, 1, 900, 0), _element(2, 1, 2, 900, 90)])
    assert evidence[1].personal_bps == 0
    assert evidence[1].raw_personal_per90 == 0.0


def test_zero_personal_minutes_uses_position_fallback(tmp_path):
    evidence = _backgrounds(tmp_path, [_element(1, 1, 1, 900, 300), _element(2, 2, 2, 0, 0),
                                       _element(3, 2, 2, 900, 180)])
    assert evidence[2].personal_minutes == 0
    assert evidence[2].fallback_scope == "POSITION"
    assert evidence[2].shrunk_per90 == pytest.approx(evidence[2].fallback_per90)


def test_position_pool_absent_uses_league_fallback(tmp_path):
    evidence = _backgrounds(tmp_path, [_element(1, 1, 2, 900, 90), _element(2, 1, 3, 0, 0)])
    assert evidence[2].fallback_scope == "LEAGUE"
    assert evidence[2].fallback_per90 == pytest.approx(90 * 90 / 900)


def test_no_league_evidence_fails(tmp_path):
    conn, _raw = _build(tmp_path, elements=[_element(1, 1, 1, 0, 0)])
    with pytest.raises(be.BackgroundAuthorityError, match="no causal league evidence"):
        be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)


def test_shrinkage_equation_by_hand():
    """w = m/(m+900); shrunk = w*raw + (1-w)*fallback."""

    minutes, bps_value, fallback = 900, 180, 45.0
    raw_per90 = bps_value * 90 / minutes          # 18.0
    weight = minutes / (minutes + be.SHRINKAGE_MINUTES)   # 0.5
    assert weight * raw_per90 + (1 - weight) * fallback == pytest.approx(31.5)


def test_shrinkage_identity_holds_on_a_loaded_population(tmp_path):
    """The hand equation must be the SAME equation the loader applies."""

    evidence = _backgrounds(tmp_path, [_element(1, 1, 2, 600, 300), _element(2, 1, 2, 900, 90)])
    entry = evidence[1]
    weight = 600 / (600 + be.SHRINKAGE_MINUTES)
    expected = weight * (300 * 90 / 600) + (1 - weight) * entry.fallback_per90
    assert entry.shrunk_per90 == pytest.approx(expected)


def test_position_comes_from_the_historical_snapshot_raw_json(tmp_path):
    evidence = _backgrounds(tmp_path, [_element(1, 1, 4, 900, 90)])
    assert evidence[1].position == "FWD"


def test_current_position_cannot_override_historical_position(tmp_path):
    conn, _raw = _build(tmp_path, elements=[_element(1, 1, 4, 900, 90)],
                        current_teams={1: 1})
    conn.execute("UPDATE players SET element_type = 1 WHERE id = 1")  # current says GKP
    conn.commit()
    evidence = be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)[0]
    assert evidence[1].position == "FWD", "historical element_type is authoritative"


# ---------------------------------------------------------------------------
# OUTCOMES — canonical placeholder semantics
# ---------------------------------------------------------------------------


def test_canonical_placeholder_is_excluded(tmp_path):
    conn, _raw = _build(tmp_path)
    signature = dict(placeholder_row()) | {"source": "element_summary"}
    assert repositories.row_is_scheduled_placeholder(signature), (
        "the fixture must actually exercise the canonical placeholder signature")
    _insert_gameweek(conn, 1, 100, placeholder_row(), source="element_summary")
    outcomes, excluded = be.final_outcomes(conn, event=TARGET_EVENT)
    assert outcomes == {} and excluded["placeholder"] == 1
    assert excluded["missing_bonus"] == 0


def test_null_minutes_placeholder_is_counted(tmp_path):
    """A placeholder whose ``minutes`` is NULL must be COUNTED, not dropped by a pre-filter.

    The old SQL pre-filter required ``minutes IS NOT NULL`` (or a non-NULL performance
    column), so this row was removed before the canonical predicate could ever see it and the
    placeholder count silently lost it.
    """

    conn, _raw = _build(tmp_path)
    null_minutes = dict(placeholder_row()) | {"minutes": None}
    assert all(value is None for value in null_minutes.values()), "truly no evidence at all"
    assert repositories.row_is_scheduled_placeholder(
        dict(null_minutes) | {"source": "element_summary"})
    _insert_gameweek(conn, 1, 100, null_minutes, source="element_summary")
    outcomes, excluded = be.final_outcomes(conn, event=TARGET_EVENT)
    assert outcomes == {}
    assert excluded["placeholder"] == 1
    assert excluded["missing_bonus"] == 0


def test_genuine_dnp_is_a_valid_realised_zero(tmp_path):
    conn, _raw = _build(tmp_path)
    completed = performance_zeroes()          # minutes = starts = bonus = bps = 0 and the rest 0
    assert not repositories.row_is_scheduled_placeholder(
        dict(completed) | {"player_id": 1, "source": "element_summary"})
    _insert_gameweek(conn, 1, 100, completed)
    outcomes, excluded = be.final_outcomes(conn, event=TARGET_EVENT)
    assert excluded["placeholder"] == 0
    assert (1, 100) in outcomes and outcomes[(1, 100)]["bonus"] == 0


def test_missing_bonus_on_a_real_row_is_excluded(tmp_path):
    """A NULL bonus on a row that carries real performance evidence is NOT a placeholder."""

    conn, _raw = _build(tmp_path)
    _insert_gameweek(conn, 1, 100,
                     performance_zeroes() | {"minutes": 90, "starts": 1, "bonus": None, "bps": 12})
    outcomes, excluded = be.final_outcomes(conn, event=TARGET_EVENT)
    assert (1, 100) not in outcomes
    assert excluded == {"placeholder": 0, "missing_bonus": 1}


def test_unfinished_event_yields_no_outcomes(tmp_path):
    conn, _raw = _build(tmp_path)
    _insert_gameweek(conn, 1, 100, performance_zeroes() | {"bonus": 1})
    conn.execute("UPDATE events SET data_checked = 0 WHERE id = ?", (TARGET_EVENT,))
    conn.commit()
    outcomes, excluded = be.final_outcomes(conn, event=TARGET_EVENT)
    assert outcomes == {} and excluded == {"placeholder": 0, "missing_bonus": 0}


# ---------------------------------------------------------------------------
# LEAKAGE — the background must be causal
# ---------------------------------------------------------------------------


def test_current_gameweeks_cannot_alter_the_snapshot_background(tmp_path):
    conn, _raw = _build(tmp_path)
    before = be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)[0]
    _insert_gameweek(conn, 1, 100, performance_zeroes() | {"minutes": 9000, "bps": 99999})
    after = be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)[0]
    assert before[1].shrunk_per90 == after[1].shrunk_per90
    assert before[1].personal_bps == after[1].personal_bps == 300


def test_a_later_snapshot_fetch_cannot_alter_the_run_7_background(tmp_path):
    conn, _raw = _build(tmp_path)
    before = be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)[0]
    later = [_element(1, 1, 1, 1800, 700)]
    _write_raw(_raw, 9, later, finished_at="2026-09-12T18:00:00Z")
    _insert_fetch_run(conn, 9, "2026-09-12T18:00:00Z", "2026-09-12T18:00:01Z", "success", _raw)
    _insert_snapshots(conn, 9, later, "2026-09-12T18:00:00Z")
    conn.commit()
    after = be.load_snapshot_backgrounds(conn, fetch_run_id=FETCH_ID)[0]
    assert before[1].shrunk_per90 == after[1].shrunk_per90
    assert after[1].personal_minutes == 900


def test_selected_historical_bps_change_DOES_alter_the_background(tmp_path):
    """Negative control: the instrument must be able to see a real change."""

    before = _backgrounds(tmp_path, [_element(1, 1, 1, 900, 300)])
    after = _backgrounds(tmp_path, [_element(1, 1, 1, 900, 600)])
    assert before[1].shrunk_per90 != after[1].shrunk_per90
    assert after[1].shrunk_per90 == pytest.approx(before[1].shrunk_per90 * 2)


# ---------------------------------------------------------------------------
# ORCHESTRATOR — the whole fixture competes in every world
# ---------------------------------------------------------------------------


def _world(pid: int, *, minutes=90, goals=0, assists=0, clean_sheets=0, conceded=0, saves=0,
           yellow=0, position="MID") -> dict:
    return {"position": position, "minutes": minutes, "goals_scored": goals,
            "assists": assists, "clean_sheets": clean_sheets, "goals_conceded": conceded,
            "saves": saves, "yellow_cards": yellow}


def _background(pid: int, per90: float, *, position="MID") -> be.BackgroundEvidence:
    return be.BackgroundEvidence(
        player_id=int(pid), position=position, personal_minutes=900, personal_bps=100,
        raw_personal_per90=per90, fallback_scope="POSITION", fallback_population_players=1,
        fallback_population_minutes=900, fallback_per90=per90,
        shrinkage_minutes=be.SHRINKAGE_MINUTES, shrunk_per90=per90)


def test_missing_world_player_raises(tmp_path):
    backgrounds = {1: _background(1, 10.0), 2: _background(2, 20.0)}
    worlds = [{1: _world(1), 2: _world(2)}, {1: _world(1)}]
    with pytest.raises(be.CaptureContractError, match="no record for"):
        be.evaluate_fixture_bonus_worlds(fixture_id=100, captured_worlds=worlds,
                                         backgrounds=backgrounds)


def test_float_ordering_breaks_a_near_tie():
    backgrounds = {1: _background(1, 10.0), 2: _background(2, 10.0000001)}
    predictions, _diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=[{1: _world(1), 2: _world(2)}], backgrounds=backgrounds)
    by_player = {p.player_id: p for p in predictions}
    assert by_player[2].expected_bonus == 3.0, "the strictly higher proxy takes 3"
    assert by_player[1].expected_bonus == 2.0


def test_exact_tie_shares_the_top_bonus():
    backgrounds = {1: _background(1, 10.0), 2: _background(2, 10.0)}
    predictions, _diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=[{1: _world(1), 2: _world(2)}], backgrounds=backgrounds)
    assert {p.expected_bonus for p in predictions} == {3.0}, "equal proxies share bonus"
    assert {p.p_bonus_3 for p in predictions} == {1.0}


def test_centring_identity_holds_inside_the_orchestrator():
    """The proxy mean must EQUAL the background level: simulated events only deviate."""

    backgrounds = {1: _background(1, 12.0), 2: _background(2, 30.0)}
    worlds = [{1: _world(1, goals=1), 2: _world(2)},
              {1: _world(1), 2: _world(2, assists=1)},
              {1: _world(1), 2: _world(2)}]
    predictions, _diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=worlds, backgrounds=backgrounds)
    by_player = {p.player_id: p for p in predictions}
    assert by_player[1].mean_bps_proxy == pytest.approx(12.0)
    assert by_player[2].mean_bps_proxy == pytest.approx(30.0)


def test_within_world_ranking_is_whole_fixture():
    """Ranking is computed across the WHOLE fixture, so a third player can take a slot."""

    backgrounds = {1: _background(1, 10.0), 2: _background(2, 20.0), 3: _background(3, 30.0)}
    worlds = [{1: _world(1), 2: _world(2), 3: _world(3)}]
    predictions, _diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=worlds, backgrounds=backgrounds)
    by_player = {p.player_id: p for p in predictions}
    assert [by_player[p].expected_bonus for p in (3, 2, 1)] == [3.0, 2.0, 1.0]


def _proxies(backgrounds, worlds):
    """The per-player proxy the orchestrator actually computes, for one fixture."""

    predictions, _diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=worlds, backgrounds=backgrounds)
    return {p.player_id: p.mean_bps_proxy for p in predictions}


def test_rival_dependence_is_proven_on_a_real_fixture():
    """My proxy is untouched while a rival's changes, and my bonus moves."""

    # Player 1 is the TOP proxy, so raising a lower rival ABOVE him must cost him a bonus.
    backgrounds = {1: _background(1, 30.0), 2: _background(2, 10.0), 3: _background(3, 20.0)}
    worlds = [{1: _world(1), 2: _world(2), 3: _world(3)}]
    _predictions, diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=worlds, backgrounds=backgrounds)
    assert diagnostics["rival_dependence_example"] is not None, (
        "the orchestrator must be able to PROVE rival dependence")
    example = diagnostics["rival_dependence_example"]
    assert example["player_bonus_before"] != example["player_bonus_after"]

    # Independent confirmation against the canonical allocator on the same proxies.
    proxies = _proxies(backgrounds, worlds)
    me = example["player_id"]
    shifted = dict(proxies)
    shifted[example["rival_id"]] = shifted[example["rival_id"]] + 1000.0
    assert ba.allocate_fixture_bonus(shifted)[me] != ba.allocate_fixture_bonus(proxies)[me]
    assert shifted[me] == proxies[me], "my own proxy never moved"


def test_limitation_flags_are_reported_per_player():
    backgrounds = {1: _background(1, 10.0, position="GKP"), 2: _background(2, 20.0)}
    worlds = [{1: _world(1, position="GKP"), 2: _world(2)}]
    predictions, diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=worlds, backgrounds=backgrounds)
    gkp = next(p for p in predictions if p.player_id == 1)
    assert gkp.limitation_flags, "an unsupported rule row must surface as a flag"
    assert set(gkp.limitation_flags) <= set(diagnostics["limitation_flags_union"])


def test_missing_background_fails_closed(tmp_path):
    with pytest.raises(be.BackgroundEvidenceError, match="no causal background for player 2"):
        be.evaluate_fixture_bonus_worlds(
            fixture_id=100, captured_worlds=[{1: _world(1), 2: _world(2)}],
            backgrounds={1: _background(1, 10.0)})


def test_no_worlds_returns_empty():
    predictions, diagnostics = be.evaluate_fixture_bonus_worlds(
        fixture_id=100, captured_worlds=[], backgrounds={})
    assert predictions == [] and diagnostics["fixtures"] == 0


def test_centred_bps_mean_equals_the_background():
    series = bc.centred_world_bps(7.5, [1.0, 2.0, 12.0])
    assert sum(series) / len(series) == pytest.approx(7.5)


def test_structural_bps_missing_event_raises():
    from fpl_brain import bps_rules as bps
    with pytest.raises(bps.BPSPrimitiveMissing):
        bc.structural_world_bps(1, "MID", {"minutes": 90})


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------


class _Player:
    def __init__(self, pid, position, bonus=1.0, p_any=0.4):
        self.player_id, self.fixture_id, self.position = pid, 100, position
        self.expected_bonus, self.p_bonus_any = bonus, p_any
        self.p_bonus_2plus = self.p_bonus_3 = 0.0
        self.mean_bps_proxy = 10.0


def test_metric_status_and_n_are_preserved():
    report = be.score_predictions([_Player(1, "DEF"), _Player(2, "DEF")],
                                  {(1, 100): {"bonus": 2}, (2, 100): {"bonus": 0}})
    assert report["status"] == "OK" and report["n"] == 2
    for key in ("bias", "mae", "brier_p_any"):
        assert report[key]["status"] == "OK"
        assert report[key]["value"] is not None
    assert report["by_position"]["FWD"]["status"] == "NO_SAMPLE"
    assert report["by_position"]["FWD"]["n"] == 0


def test_missing_outcome_is_excluded_and_counted():
    report = be.score_predictions([_Player(1, "DEF"), _Player(2, "DEF")],
                                  {(1, 100): {"bonus": 2}})
    assert report["n"] == 1
    assert report["excluded"] == {"missing_outcome": 1}


def test_no_eligible_prediction_reports_no_sample():
    report = be.score_predictions([_Player(1, "DEF")], {})
    assert report == {"status": "NO_SAMPLE", "n": 0,
                      "excluded": {"missing_outcome": 1}}


def test_bias_sign_is_overprediction():
    report = be.score_predictions([_Player(1, "MID", bonus=2.0)],
                                  {(1, 100): {"bonus": 1}})
    assert report["bias"]["value"] == pytest.approx(1.0)


def test_population_digest_is_deterministic_and_order_independent():
    a = be.population_digest([(2, 100), (1, 100)])
    b = be.population_digest([(1, 100), (2, 100)])
    assert a == b and a.startswith("sha256:")


def test_population_digest_distinguishes_different_populations():
    assert be.population_digest([(1, 100)]) != be.population_digest([(1, 101)])


# ---------------------------------------------------------------------------
# CHAIN / PROVENANCE
# ---------------------------------------------------------------------------


_UNSET = object()


def _add_chain(conn, *, as_of=CUTOFF, fetch_run_id=FETCH_ID, status="complete",
               families=None, official_fetch=None, official_run_ids=_UNSET,
               without_provenance_for=None):
    """Write the four frozen predictive runs, with per-role control over the provenance field.

    ``official_run_ids`` overrides the record for EVERY role; ``without_provenance_for`` names
    ONE role whose record is written as SQL NULL, which proves the per-role loop rather than a
    single global check.
    """

    families = families or _CHAIN_FAMILIES
    default = json.dumps({"fetch": {"run_id": int(
        fetch_run_id if official_fetch is None else official_fetch)}})
    for role, run_id in CHAIN.items():
        recorded = default if official_run_ids is _UNSET else official_run_ids
        if role == without_provenance_for:
            recorded = None
        conn.execute(
            "INSERT OR REPLACE INTO projection_runs (id, model_family, model_version,"
            " generated_at, planning_event, data_cutoff, official_run_ids, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (run_id, families[role], "v1.0.0", as_of, TARGET_EVENT, as_of, recorded, status))
    conn.commit()


def _add_xpts_rows(conn, keys, *, bonus=0.05, minutes_run=None, team_run=None, rate_run=None,
                   run_id=None):
    # OR IGNORE: when a test also declares the full chain, that richer row must survive.
    conn.execute(
        "INSERT OR IGNORE INTO projection_runs (id, model_family, model_version,"
        " generated_at, planning_event, data_cutoff, status) VALUES (?,?,?,?,?,?,?)",
        (int(run_id or CHAIN["xpts"]), "xpts_v1", "v1.0.0", CUTOFF, TARGET_EVENT, CUTOFF,
         "complete"))
    for player_id, fixture_id in keys:
        conn.execute(
            "INSERT OR REPLACE INTO player_fixture_xpts_projections (projection_run_id,"
            " player_id, fixture_id, event, team_id, opponent_id, position, minutes_run_id,"
            " team_run_id, rate_run_id, payload_json, model_version, scoring_rules_version,"
            " generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (int(run_id or CHAIN["xpts"]), int(player_id), int(fixture_id), TARGET_EVENT, 1, 2,
             "MID", int(minutes_run or CHAIN["minutes"]), int(team_run or CHAIN["team"]),
             int(rate_run or CHAIN["rate"]), json.dumps({"bonus_xpts": float(bonus)}),
             "v1.0.0", "v1.0.0", CUTOFF))
    conn.commit()


def test_incoherent_chain_is_refused(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn)
    _add_xpts_rows(conn, [(1, 100)], minutes_run=999)
    with pytest.raises(be.BackgroundAuthorityError, match="MINUTES_RUN_MISMATCH"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_cutoff_mismatch_in_the_chain_is_refused(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, as_of="2026-09-12T09:00:00Z")
    with pytest.raises(be.BackgroundAuthorityError, match="cutoff"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_post_deadline_chain_is_refused(tmp_path):
    conn, raw_dir = _build(tmp_path)
    late = "2026-09-12T13:00:00Z"
    _add_chain(conn, as_of=late)
    with pytest.raises(be.BackgroundAuthorityError, match="not before the deadline"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=late, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_incomplete_chain_run_is_refused(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, status="running")
    with pytest.raises(be.BackgroundAuthorityError, match="not complete"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_wrong_chain_family_is_refused(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, families=dict(_CHAIN_FAMILIES, minutes="xpts_v1"))
    with pytest.raises(be.BackgroundAuthorityError, match="expected 'minutes_v1'"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


@pytest.mark.parametrize("role", ["xpts", "minutes", "team", "rate"])
def test_missing_official_run_ids_fails_closed(tmp_path, role):
    """§4: a chain run with NO provenance record cannot be shown to have used this fetch."""

    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, without_provenance_for=role)
    with pytest.raises(be.BackgroundAuthorityError, match="records no official_run_ids"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_official_run_ids_without_a_fetch_run_id_fails_closed(tmp_path):
    """A record that exists but names no fetch proves nothing, so it must not pass."""

    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, official_run_ids=json.dumps({"minutes": {"run_id": 127}}))
    with pytest.raises(be.BackgroundAuthorityError, match="records no fetch.run_id"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_official_run_ids_with_an_empty_fetch_fails_closed(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, official_run_ids=json.dumps({"fetch": {}}))
    with pytest.raises(be.BackgroundAuthorityError, match="records no fetch.run_id"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_unparseable_official_run_ids_fails_closed(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, official_run_ids="{not json")
    with pytest.raises(be.BackgroundAuthorityError, match="official_run_ids is unparseable"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_non_integer_fetch_run_id_fails_closed(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, official_run_ids=json.dumps({"fetch": {"run_id": "thirty-six"}}))
    with pytest.raises(be.BackgroundAuthorityError, match="is not an integer"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_official_fetch_mismatch_in_the_chain_is_refused(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn, official_fetch=99)
    with pytest.raises(be.BackgroundAuthorityError, match="records official fetch 99"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=CHAIN["xpts"])


def test_soft_baseline_must_be_the_chain_xpts_run(tmp_path):
    conn, raw_dir = _build(tmp_path)
    _add_chain(conn)
    _add_xpts_rows(conn, [(1, 100)])
    with pytest.raises(be.BackgroundAuthorityError, match="is not the chain's xpts run"):
        be.evaluate_event(conn, event=TARGET_EVENT, as_of=CUTOFF, deadline=DEADLINE, chain=CHAIN,
                          fetch_run_id=FETCH_ID, raw_dir=raw_dir, target_event=TARGET_EVENT,
                          xpts_run_id=140)


def test_calibration_identity_must_be_present_never_a_silent_none(monkeypatch):
    """§11: the replay identity must carry the calibration state, or refuse to emit."""

    from fpl_brain import monte_carlo as mc_mod
    config = mc_mod.MonteCarloConfig(simulations=10, occupancy_audit=False)

    identity = be._calibration_state_identity(config)
    assert identity.startswith("sha256:"), "a real calibration state identity is required"

    def _empty(_config):
        return {}

    monkeypatch.setattr(mc_mod, "calibration_provenance", _empty, raising=False)
    with pytest.raises(be.CaptureContractError, match="no calibration_state_identity"):
        be._calibration_state_identity(config)


def test_calibration_identity_is_unavailable_without_the_provenance_hook(monkeypatch):
    """The hook itself must exist: a missing function is not a licence to emit None."""

    from fpl_brain import monte_carlo as mc_mod
    config = mc_mod.MonteCarloConfig(simulations=10, occupancy_audit=False)
    monkeypatch.delattr(mc_mod, "calibration_provenance", raising=False)
    with pytest.raises(be.CaptureContractError, match="calibration_provenance is unavailable"):
        be._calibration_state_identity(config)


def test_legacy_authority_never_backfills_a_generation_row(tmp_path):
    """§14/§16: the retrospective classification must not mutate the acceptance table."""

    conn, raw_dir = _build(tmp_path)
    before = conn.execute("SELECT COUNT(*) FROM bootstrap_generations").fetchone()[0]
    evidence = _verify(conn, raw_dir)
    after = conn.execute("SELECT COUNT(*) FROM bootstrap_generations").fetchone()[0]
    assert before == after == 0
    assert evidence.formal_generation_record == "NONE"
    assert evidence.as_dict()["generation_accepted"] is None


def test_soft_baseline_returns_incomplete_when_a_key_is_missing(tmp_path):
    conn, _raw = _build(tmp_path)
    _add_xpts_rows(conn, [(1, 100)])
    report = be.soft_baseline(conn, xpts_run_id=CHAIN["xpts"], keys=[(1, 100), (2, 100)],
                              outcomes={(1, 100): {"bonus": 0}, (2, 100): {"bonus": 0}})
    assert report["status"] == "INCOMPLETE_BASELINE"
    assert report["missing"] == {"player_id": 2, "fixture_id": 100}


def test_soft_baseline_population_digest_matches_the_structural_one(tmp_path):
    conn, _raw = _build(tmp_path)
    keys = [(1, 100), (2, 100)]
    _add_xpts_rows(conn, keys, bonus=0.25)
    outcomes = {(pid, 100): {"bonus": 1} for pid, _ in keys}
    report = be.soft_baseline(conn, xpts_run_id=CHAIN["xpts"], keys=keys, outcomes=outcomes)
    assert report["status"] == "OK" and report["n"] == 2
    assert report["population_digest"] == be.population_digest(keys)
    assert report["mean_predicted"] == pytest.approx(0.25)
    assert report["brier"] == "NOT_APPLICABLE — expected points is not a probability"
