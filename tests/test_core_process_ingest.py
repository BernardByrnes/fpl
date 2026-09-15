"""FPL Core Insights ingestion + feature lab: hashing, immutability, crosswalks,
EPL-only enforcement, placeholder exclusion, point-in-time safety, determinism.

Everything here is synthetic — no network, no provider files — so the contract is
pinned independently of the cached data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl_brain import core_process_features as cf
from fpl_brain import core_process_ingest as ci

COMMIT = "a" * 40
TOURNAMENT = "Premier League"


class FakeTree:
    """An in-memory source repository."""

    def __init__(self, files: dict[str, bytes]):
        self.files = dict(files)

    def exists(self, relative_path: str) -> bool:
        return relative_path in self.files

    def read_bytes(self, relative_path: str) -> bytes:
        if relative_path not in self.files:
            raise ci.CoreProcessError(
                f"{ci.DIAG_SOURCE_FILE_MISSING}: {relative_path}", reasons=[ci.DIAG_SOURCE_FILE_MISSING]
            )
        return self.files[relative_path]


def _csv(header: str, rows: list[str]) -> bytes:
    return (header + "\n" + "\n".join(rows) + "\n").encode("utf-8")


MATCH_FILE = _csv(
    "gameweek,kickoff_time,home_team,away_team,home_score,away_score,finished,match_id,tournament",
    ["1,2026-08-21T19:00:00,3.0,9.0,3.0,0.0,True,26-27-prem-arsenal-vs-coventry-city,prem"],
)
PLAYERS_FILE = _csv(
    "player_code,player_id,first_name,second_name,web_name,team_code,position",
    ["208706,452,Bruno,Guimaraes,Bruno G.,3,Midfielder", "569577,606,Triston,Rowe,Rowe,7,Defender"],
)


def _tree(**overrides) -> FakeTree:
    files = {
        ci.gameweek_path(season="2026-2027", tournament=TOURNAMENT, gameweek=1, name="matches.csv"): MATCH_FILE,
        ci.gameweek_path(season="2026-2027", tournament=TOURNAMENT, gameweek=1, name="players.csv"): PLAYERS_FILE,
    }
    files.update(overrides)
    return FakeTree(files)


def _cache(tmp_path) -> ci.CoreProcessCache:
    return ci.CoreProcessCache(tmp_path, repository_commit_sha=COMMIT, retrieved_at="2026-09-15T05:00:00Z")


# ---------------------------------------------------------------------------
# §20 source hashing + immutable versioning
# ---------------------------------------------------------------------------


def test_source_hashing_covers_bytes_and_schema(tmp_path):
    tree = _tree()
    cache = _cache(tmp_path)
    record = cache.cache_file(tree, season="2026-2027", gameweek=1, name="matches.csv")
    assert record.sha256 == ci.sha256_bytes(MATCH_FILE)
    assert record.size_bytes == len(MATCH_FILE)
    assert record.row_count == 1
    assert record.schema_hash == ci.schema_hash(MATCH_FILE)
    # the schema hash is the HEADER, so a data-only change does not move it
    changed = MATCH_FILE.replace(b"3.0,0.0", b"2.0,1.0")
    assert ci.schema_hash(changed) == record.schema_hash
    assert ci.sha256_bytes(changed) != record.sha256
    assert Path(record.cache_path).read_bytes() == MATCH_FILE


def test_recaching_identical_bytes_is_idempotent(tmp_path):
    cache = _cache(tmp_path)
    first = cache.cache_file(_tree(), season="2026-2027", gameweek=1, name="matches.csv")
    second = cache.cache_file(_tree(), season="2026-2027", gameweek=1, name="matches.csv")
    assert first.version == second.version == 1
    assert first.sha256 == second.sha256


def test_a_provider_correction_creates_a_new_version(tmp_path):
    cache = _cache(tmp_path)
    corrected = MATCH_FILE.replace(b"3.0,0.0,True", b"2.0,2.0,True")
    first = cache.cache_file(_tree(), season="2026-2027", gameweek=1, name="matches.csv")
    cache.write_manifest()
    second = cache.cache_file(
        _tree(**{ci.gameweek_path(season="2026-2027", tournament=TOURNAMENT, gameweek=1, name="matches.csv"): corrected}),
        season="2026-2027", gameweek=1, name="matches.csv",
    )
    assert second.version == 2
    assert second.sha256 != first.sha256
    # the ORIGINAL bytes survive untouched
    assert Path(first.cache_path).read_bytes() == MATCH_FILE
    assert Path(second.cache_path).read_bytes() == corrected
    cache.write_manifest()
    manifest = cache.manifest()
    assert sorted(r["version"] for r in manifest) == [1, 2]
    assert len({r["sha256"] for r in manifest}) == 2


def test_manifest_is_append_only(tmp_path):
    cache = _cache(tmp_path)
    cache.cache_file(_tree(), season="2026-2027", gameweek=1, name="matches.csv")
    cache.write_manifest()
    first = cache.manifest()
    cache.write_manifest()  # second run, no new files
    assert cache.manifest() == first


# ---------------------------------------------------------------------------
# §20 EPL-only filtering / Champions League leakage
# ---------------------------------------------------------------------------


def test_a_non_premier_league_row_fails_closed(tmp_path):
    leaked = _csv(
        "gameweek,kickoff_time,home_team,away_team,home_score,away_score,finished,match_id,tournament",
        [
            "1,2026-08-21T19:00:00,3.0,9.0,3.0,0.0,True,26-27-prem-arsenal-vs-coventry-city,prem",
            "1,2026-09-16T19:00:00,3.0,57.0,1.0,0.0,True,26-27-ucl-arsenal-vs-bayern,ucl",
        ],
    )
    cache = _cache(tmp_path)
    with pytest.raises(ci.CoreProcessError) as failure:
        cache.cache_file(
            _tree(**{ci.gameweek_path(season="2026-2027", tournament=TOURNAMENT, gameweek=1, name="matches.csv"): leaked}),
            season="2026-2027", gameweek=1, name="matches.csv",
        )
    assert ci.DIAG_TOURNAMENT_LEAK in str(failure.value)


def test_a_pure_champions_league_file_cannot_be_ingested_as_premier_league(tmp_path):
    ucl = _csv(
        "gameweek,kickoff_time,home_team,away_team,home_score,away_score,finished,match_id,tournament",
        ["1,2026-09-16T19:00:00,3.0,57.0,1.0,0.0,True,26-27-ucl-arsenal-vs-bayern,ucl"],
    )
    cache = _cache(tmp_path)
    with pytest.raises(ci.CoreProcessError) as failure:
        cache.cache_file(
            _tree(**{ci.gameweek_path(season="2026-2027", tournament=TOURNAMENT, gameweek=1, name="matches.csv"): ucl}),
            season="2026-2027", gameweek=1, name="matches.csv",
        )
    assert ci.DIAG_TOURNAMENT_LEAK in str(failure.value)


def test_the_canonical_path_is_the_tournament_scoped_one():
    assert ci.gameweek_path(season="2026-2027", tournament=TOURNAMENT, gameweek=3, name="shots.csv") == (
        "data/2026-2027/By Tournament/Premier League/GW3/shots.csv"
    )
    assert ci.PREMIER_LEAGUE_TOURNAMENT_CODE == "prem"


# ---------------------------------------------------------------------------
# §20 missing source file fails closed
# ---------------------------------------------------------------------------


def test_a_missing_source_file_fails_closed(tmp_path):
    cache = _cache(tmp_path)
    with pytest.raises(ci.CoreProcessError) as failure:
        cache.cache_file(_tree(), season="2026-2027", gameweek=1, name="shots.csv")
    assert ci.DIAG_SOURCE_FILE_MISSING in str(failure.value)


# ---------------------------------------------------------------------------
# §20 player crosswalk
# ---------------------------------------------------------------------------


OFFICIAL_PLAYERS = [
    {"id": 452, "code": 208706, "web_name": "Bruno G."},
    {"id": 606, "code": 569577, "web_name": "Rowe"},
]
EXTERNAL_PLAYERS = [
    {"player_id": "452", "player_code": "208706", "web_name": "Bruno G."},
    {"player_id": "606", "player_code": "569577", "web_name": "Rowe"},
]


def test_player_crosswalk_matches_on_both_keys():
    crosswalk = ci.build_player_crosswalk(OFFICIAL_PLAYERS, EXTERNAL_PLAYERS)
    assert crosswalk.ok
    assert crosswalk.id_matches == 2 and crosswalk.code_matches == 2 and crosswalk.both_keys_agree == 2
    assert crosswalk.official_id_for(606) == 606
    assert crosswalk.ambiguous == () and crosswalk.official_only == ()


def test_player_crosswalk_flags_a_disagreeing_key_instead_of_guessing():
    """The two keys point at different official players -> ambiguous, not a match."""

    external = [{"player_id": "452", "player_code": "569577", "web_name": "Bruno G."}]
    crosswalk = ci.build_player_crosswalk(OFFICIAL_PLAYERS, external)
    assert crosswalk.ambiguous == (452,)
    assert not crosswalk.ok
    assert crosswalk.official_id_for(452) is None


def test_player_crosswalk_reports_external_only_and_official_only():
    external = list(EXTERNAL_PLAYERS) + [{"player_id": "999", "player_code": "1", "web_name": "Nobody"}]
    crosswalk = ci.build_player_crosswalk(OFFICIAL_PLAYERS + [{"id": 7, "code": 77, "web_name": "Unlisted"}], external)
    assert crosswalk.external_only == (999,)
    assert crosswalk.official_only == (7,)
    assert not crosswalk.ok


def test_player_crosswalk_never_uses_names_as_the_join():
    """A name that matches but whose ids do not must NOT produce a match."""

    # neither id nor code resolves, but the NAME matches a real player
    external = [{"player_id": "888", "player_code": "999999", "web_name": "Bruno G."}]
    crosswalk = ci.build_player_crosswalk(OFFICIAL_PLAYERS, external)
    assert crosswalk.entries == ()
    assert crosswalk.external_only == (888,)
    assert crosswalk.name_only_matches == 1     # reported, never used


# ---------------------------------------------------------------------------
# §20 fixture crosswalk
# ---------------------------------------------------------------------------

OFFICIAL_TEAMS = [{"id": 1, "code": 3}, {"id": 2, "code": 9}, {"id": 3, "code": 7}]
OFFICIAL_FIXTURES = [
    {"id": 10, "event": 1, "team_h": 1, "team_a": 2, "kickoff_time": "2026-08-21T19:00:00Z"},
]


def test_fixture_crosswalk_is_one_to_one():
    matches = [{"match_id": "m1", "gameweek": "1", "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3.0", "away_team": "9.0", "tournament": "prem"}]
    crosswalk = ci.build_fixture_crosswalk(OFFICIAL_FIXTURES, matches, OFFICIAL_TEAMS)
    assert crosswalk.ok and crosswalk.one_to_one and crosswalk.matched == 1
    assert crosswalk.entries[0].official_fixture_id == 10
    assert crosswalk.entries[0].event == 1


def test_fixture_crosswalk_normalises_kickoff_timezones():
    matches = [{"match_id": "m1", "gameweek": 1, "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3", "away_team": "9", "tournament": "prem"}]
    crosswalk = ci.build_fixture_crosswalk(OFFICIAL_FIXTURES, matches, OFFICIAL_TEAMS)
    assert crosswalk.matched == 1          # 'Z' vs naive compare equal
    assert ci.normalise_kickoff("2026-08-21T19:00:00Z") == ci.normalise_kickoff("2026-08-21T19:00:00")


def test_fixture_crosswalk_fails_closed_on_an_unknown_team_code():
    matches = [{"match_id": "m1", "gameweek": 1, "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3", "away_team": "999", "tournament": "prem"}]
    crosswalk = ci.build_fixture_crosswalk(OFFICIAL_FIXTURES, matches, OFFICIAL_TEAMS)
    assert not crosswalk.ok
    assert crosswalk.unmapped_team_codes == (999,)
    assert crosswalk.unmatched == ("m1",)


def test_fixture_crosswalk_refuses_a_duplicated_official_fixture():
    duplicated = OFFICIAL_FIXTURES + [{"id": 11, "event": 1, "team_h": 1, "team_a": 2,
                                       "kickoff_time": "2026-08-21T19:00:00Z"}]
    matches = [{"match_id": "m1", "gameweek": 1, "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3", "away_team": "9", "tournament": "prem"}]
    crosswalk = ci.build_fixture_crosswalk(duplicated, matches, OFFICIAL_TEAMS)
    assert crosswalk.ambiguous == ("m1",) and not crosswalk.ok


def test_fixture_crosswalk_refuses_a_gameweek_mismatch():
    matches = [{"match_id": "m1", "gameweek": 2, "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3", "away_team": "9", "tournament": "prem"}]
    crosswalk = ci.build_fixture_crosswalk(OFFICIAL_FIXTURES, matches, OFFICIAL_TEAMS)
    assert crosswalk.ambiguous == ("m1",)


# ---------------------------------------------------------------------------
# §20 point-in-time safety
# ---------------------------------------------------------------------------


def test_snapshot_availability_uses_retrieval_time_only(tmp_path):
    cache = ci.CoreProcessCache(tmp_path, repository_commit_sha=COMMIT, retrieved_at="2026-09-15T05:00:00Z")
    record = cache.cache_file(_tree(), season="2026-2027", gameweek=1, name="matches.csv")
    assert record.available_at("2026-09-15T05:00:00Z") is True
    assert record.available_at("2026-09-15T04:59:59Z") is False
    assert record.available_at(None) is True


def test_a_later_retrieval_cannot_leak_into_an_earlier_cutoff(tmp_path):
    """Post-match data retrieved after a cutoff is invisible to that cutoff."""

    cache = ci.CoreProcessCache(tmp_path, repository_commit_sha=COMMIT, retrieved_at="2026-09-15T05:00:00Z")
    cache.cache_file(_tree(), season="2026-2027", gameweek=1, name="matches.csv")
    cache.write_manifest()
    before = cache.available_records(season="2026-2027", cutoff="2026-09-14T00:00:00Z")
    after = cache.available_records(season="2026-2027", cutoff="2026-09-16T00:00:00Z")
    assert before == [] and len(after) == 1
    # the KICKOFF is irrelevant to availability
    assert after[0]["retrieved_at"] > "2026-08-21T19:00:00Z"


# ---------------------------------------------------------------------------
# §20 placeholders, determinism, shrinkage
# ---------------------------------------------------------------------------

HEADER = ("player_id,match_id,minutes_played,goals,assists,total_shots,xg,xa,shots_on_target,"
          "chances_created,touches_opposition_box,tackles_won,interceptions,recoveries,blocks,"
          "clearances,duels_won,aerial_duels_won,start_min,finish_min,defensive_contributions")


def _pms_rows() -> list[dict[str, str]]:
    import csv
    import io
    payload = _csv(HEADER, [
        "452,m1,90,1,0,4,0.8,0.2,2,3,6,2,1,7,0,2,5,2,0,90,",
        "606,m1,0,,,,,,,,,,,,,,,,,",                     # 0-minute placeholder
        "606,m2,0,,,,,,,,,,,,,,,,,",                     # another placeholder
    ])
    return [dict(r) for r in csv.DictReader(io.StringIO(payload.decode()))]


def _crosswalks():
    players = ci.build_player_crosswalk(OFFICIAL_PLAYERS, EXTERNAL_PLAYERS)
    matches = [{"match_id": "m1", "gameweek": 1, "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3.0", "away_team": "9.0", "tournament": "prem"},
               {"match_id": "m2", "gameweek": 1, "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3.0", "away_team": "9.0", "tournament": "prem"}]
    return players, ci.build_fixture_crosswalk(OFFICIAL_FIXTURES, matches[:1], OFFICIAL_TEAMS)


def test_zero_minute_rows_are_not_observations():
    players, fixtures = _crosswalks()
    rows = cf.build_player_match_process(_pms_rows(), player_crosswalk=players, fixture_crosswalk=fixtures)
    by_player = {}
    for row in rows:
        by_player.setdefault(row.external_player_id, []).append(row)
    placeholder = by_player[606][0]
    assert placeholder.minutes_played == 0 and placeholder.played is False
    assert placeholder.started is False
    assert placeholder.external_defensive_action_sum("MID") is None
    played = by_player[452][0]
    assert played.played is True and played.started is True


def test_attacking_features_exclude_placeholders_and_are_deterministic():
    players, fixtures = _crosswalks()
    rows = cf.build_player_match_process(_pms_rows(), player_crosswalk=players, fixture_crosswalk=fixtures)
    played = [row for row in rows if row.external_player_id == 452]
    first = cf.attacking_features(played, prior_exposure=900.0)
    second = cf.attacking_features(played, prior_exposure=900.0)
    assert first == second
    assert first.matches_used == 1 and first.minutes == 90.0
    # a single match is heavily shrunk toward the zero prior
    assert first.features["shots_per90"] == pytest.approx(
        cf.shrunken_per90(total=4.0, minutes=90.0, prior_per90=0.0, prior_minutes=900.0))
    assert first.features["shots_per90"] < 4.0
    assert first.features["shots_per90"] == pytest.approx(4.0 / 990.0 * 90.0)
    assert first.as_dict()["features"]["xg_external_per90"] == pytest.approx(0.8 / 990.0 * 90.0)


def test_shrinkage_moves_toward_the_prior():
    # per-minute rate helper: a tiny sample is dominated by the prior
    tiny = cf.shrunken_rate(total=6.0, exposure=90.0, prior_rate=0.01, prior_exposure=900.0)
    assert tiny == pytest.approx((6.0 + 0.01 * 900.0) / 990.0)
    assert abs(tiny - 6.0 / 90.0) > abs(tiny - 0.01)          # closer to the prior
    large = cf.shrunken_rate(total=600.0, exposure=9000.0, prior_rate=0.01, prior_exposure=900.0)
    assert abs(large - 600.0 / 9000.0) < abs(large - 0.01)     # large sample dominates
    assert cf.shrunken_rate(total=0.0, exposure=0.0, prior_rate=2.0, prior_exposure=0.0) == 2.0
    # ...and the PER-90 helper really expresses the feature in per-90 units
    per90 = cf.shrunken_per90(total=6.0, minutes=90.0, prior_per90=0.0, prior_minutes=900.0)
    assert per90 == pytest.approx(6.0 / 990.0 * 90.0)
    assert per90 < 6.0


def test_defcon_thresholds_and_probability_come_from_the_rules():
    assert cf.defcon_threshold("DEF") == 10
    assert cf.defcon_threshold("MID") == 12
    assert cf.defcon_threshold("FWD") == 12
    assert cf.defcon_threshold("GKP") is None
    assert cf.poisson_tail(0.0, 10) == 0.0
    assert cf.poisson_tail(50.0, 10) > 0.99
    assert 0.0 <= cf.poisson_tail(9.0, 10) <= 1.0


def test_defcon_features_use_the_position_action_set():
    players, fixtures = _crosswalks()
    rows = cf.build_player_match_process(_pms_rows(), player_crosswalk=players, fixture_crosswalk=fixtures)
    played = [row for row in rows if row.external_player_id == 452]
    defence = cf.defcon_features(played, position="DEF", expected_minutes=90.0)
    midfield = cf.defcon_features(played, position="MID", expected_minutes=90.0)
    # DEF uses CBIT (2 blocks + 1 interception + 0 recoveries + 2 tackles = 5);
    # MID uses CBIRT, which adds the 7 recoveries
    assert defence.predictors["defensive_action_rate"] is not None
    assert midfield.predictors["defensive_action_rate"] > defence.predictors["defensive_action_rate"]
    assert defence.projected_defcon is not None and defence.p_defcon_threshold is not None
    assert 0.0 <= defence.p_defcon_threshold <= 1.0
    assert defence.expected_defcon_points == pytest.approx(cf.DEFAULT_SCORING_RULES.defcon_points * defence.p_defcon_threshold)
    assert cf.defcon_features(played, position="DEF", expected_minutes=90.0) == defence


def test_high_quality_chance_proxy_is_named_and_versioned():
    shot = cf.ShotFact(
        match_id="m1", official_player_id=452, official_fixture_id=10, shot_index=1, minute=10,
        is_home=True, outcome="Goal", situation="RegularPlay", body_part="RightFoot",
        xg=0.4, xgot=0.6, start_x=90.0, start_y=40.0, goal_mouth_y=None, goal_mouth_z=None,
        goal_mouth_location="",
    )
    assert cf.high_quality_chance_proxy([shot]) == 1
    assert cf.HIGH_QUALITY_CHANCE_PROXY_VERSION.startswith("hqc_proxy_v")
    assert "big_chance" not in cf.HIGH_QUALITY_CHANCE_PROXY_VERSION


def test_team_process_exposes_the_opponent_facing_view():
    matches = [{"match_id": "m1", "gameweek": 1, "kickoff_time": "2026-08-21T19:00:00",
                "home_team": "3.0", "away_team": "9.0", "tournament": "prem",
                "home_total_shots": "10", "away_total_shots": "7",
                "home_shots_on_target": "4", "away_shots_on_target": "2",
                "home_shots_inside_box": "6", "away_shots_inside_box": "3",
                "home_touches_in_opposition_box": "20", "away_touches_in_opposition_box": "12",
                "home_non_penalty_xg": "1.1", "away_non_penalty_xg": "0.8",
                "home_expected_goals_xg": "1.2", "away_expected_goals_xg": "0.9",
                "home_big_chances": "2", "away_big_chances": "1",
                "home_possession": "60", "away_possession": "40"}]
    fixture_crosswalk = ci.build_fixture_crosswalk(OFFICIAL_FIXTURES, matches, OFFICIAL_TEAMS)
    rows = cf.build_team_match_process(matches, fixture_crosswalk=fixture_crosswalk, official_teams=OFFICIAL_TEAMS)
    assert len(rows) == 2
    home = next(row for row in rows if row.is_home)
    away = next(row for row in rows if not row.is_home)
    assert home.team_id == 1 and home.opponent_id == 2
    allowed = home.opponent_facing()
    assert allowed["opponent_shots_allowed"] == 10.0      # the official fixture's home side
    assert away.opponent_facing()["opponent_shots_allowed"] == 7.0


def test_crosswalk_report_shape_is_serialisable():
    players, fixtures = _crosswalks()
    payload = {"player": players.as_dict(), "fixture": fixtures.as_dict()}
    assert json.loads(json.dumps(payload))["fixture"]["one_to_one"] is True
