"""Minutes v1.4 substitution-coherence tests."""

from __future__ import annotations

import copy
import random
import statistics

import pytest

from fpl_brain import substitution_model as sm

CFG = sm.SubstitutionConfig()

# A complete synthetic evidence set: the league pool actually observed on GW4.
EVIDENCE = {
    "team_fixtures": 60,
    "starters_seen": 660,
    "k_counts": [0, 0, 1, 5, 18, 36],
    "k_probabilities": [0.0, 0.0, 1 / 60, 5 / 60, 18 / 60, 36 / 60],
    "entry_minutes": [72.0] * 120 + [58.0] * 20 + [30.0] * 1 + [85.0] * 128,
    "exit_minutes": [70.0] * 100 + [55.0] * 20 + [85.0] * 137,
    "gk_substitutions": 0,
    "red_card_exits": 1,
    "expected_substitutions": 4.4833,
    "evidence_source": "test",
}


def _side(n=25, *, p_start=0.6, p80=0.8, cameo=0.15, p_available=1.0):
    players = []
    starters = 11
    for i in range(n):
        if i < starters:
            players.append({"p_start": p_start, "p_available": p_available,
                            "p80_given_start": p80, "cameo_propensity": cameo,
                            "expected_minutes_if_start": 90.0})
        else:
            players.append({"p_start": 0.0, "p_available": p_available,
                            "p80_given_start": 0.0, "cameo_propensity": cameo,
                            "expected_minutes_if_start": 0.0})
    return players


def _side_eleven():
    """A side whose start mass is exactly 11."""
    players = []
    for i in range(11):
        players.append({"p_start": 1.0, "p_available": 1.0, "p80_given_start": 0.8,
                        "cameo_propensity": 0.0, "expected_minutes_if_start": 90.0})
    for i in range(14):
        players.append({"p_start": 0.0, "p_available": 1.0, "p80_given_start": 0.0,
                        "cameo_propensity": 0.15, "expected_minutes_if_start": 0.0})
    return players


def _profile(side=None, evidence=None, config=CFG, **kw):
    return sm.build_team_substitution_profile(
        side or _side_eleven(), evidence or EVIDENCE, config, fixture_id=1, team_id=1, **kw
    )


def test_substitution_count_distribution_is_valid_on_zero_to_five():
    profile = _profile()
    assert len(profile["p_sub_count"]) == sm.MAX_ORDINARY_SUBSTITUTIONS + 1
    assert all(0.0 <= p <= 1.0 for p in profile["p_sub_count"])
    assert sum(profile["p_sub_count"]) == pytest.approx(1.0, abs=1e-9)


def test_expected_event_mass_never_exceeds_the_limit():
    profile = _profile()
    assert sum(profile["event_time_masses"]) <= sm.MAX_ORDINARY_SUBSTITUTIONS + 1e-9
    assert profile["expected_substitutions"] == pytest.approx(4.4833, abs=1e-3)


def test_event_mass_above_limit_is_rejected():
    evidence = dict(EVIDENCE)
    evidence["k_probabilities"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    evidence["k_probabilities"] = [0.0] * 6
    evidence["k_probabilities"][5] = 1.0
    # six events per side is impossible under the limit
    evidence["k_probabilities"] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]  # sums to 2 -> E[K]=9
    with pytest.raises(ValueError):
        _profile(evidence=evidence)


def test_exit_mass_equals_entry_mass_per_time_band():
    side = _side_eleven()
    profile = _profile(side)
    for band_index, mass in enumerate(profile["event_time_masses"]):
        exits = sum(p["exit_mass_by_band"][band_index] for p in profile["players"])
        entries = sum(p["entry_mass_by_band"][band_index] for p in profile["players"])
        # stored masses are rounded to 9 dp, so compare at 1e-6
        assert exits == pytest.approx(mass, abs=1e-6)
        assert entries == pytest.approx(mass, abs=1e-6)


def test_player_exit_mass_never_exceeds_start_probability():
    side = _side(n=25, p_start=0.5)
    profile = _profile(side)
    for player in profile["players"]:
        assert player["exit_total"] <= player["p_start"] + 1e-9


def test_player_entry_mass_never_exceeds_non_start_available_mass():
    side = _side(n=25, p_start=0.4)
    profile = _profile(side)
    for player in profile["players"]:
        assert player["entry_total"] <= player["p_available"] - player["p_start"] + 1e-9


def test_goalkeeper_substitutions_are_excluded_when_evidence_has_none():
    profile = _profile()
    assert profile["gk_event_mass"] == 0.0
    assert "GK_SUBSTITUTION_RARE_EVENT_EXCLUDED" in profile["risk_flags"]


def test_gk_event_mass_is_used_when_evidence_has_substitutions():
    evidence = dict(EVIDENCE)
    evidence["gk_substitutions"] = 3
    profile = _profile(evidence=evidence)
    assert "GK_SUBSTITUTION_RARE_EVENT_EXCLUDED" not in profile["risk_flags"]


def test_red_card_rows_are_excluded_from_exit_inference():
    import sqlite3

    from fpl_brain.database import connect_database
    from fpl_brain.models import EventRecord, FixtureRecord, PlayerGameweekRecord, PlayerRecord, PositionRecord, TeamRecord

    conn = connect_database(":memory:")
    with conn:
        from fpl_brain import repositories as repo

        repo.upsert_teams(conn, [TeamRecord(id=1, name="A"), TeamRecord(id=2, name="B")])
        repo.upsert_positions(conn, [PositionRecord(id=p, singular_name_short=n)
                                     for p, n in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))])
        repo.upsert_players(conn, [PlayerRecord(id=i, web_name=f"P{i}", team_id=1, element_type=3)
                                   for i in range(1, 15)])
        repo.upsert_events(conn, [EventRecord(id=1, finished=1, data_checked=1, raw_json={}),
                                  EventRecord(id=2, finished=1, data_checked=1, raw_json={})])
        repo.upsert_fixtures(conn, [FixtureRecord(id=1, event=1, team_h=1, team_a=2, finished=1, started=1,
                                                  kickoff_time="2026-08-22T14:00:00Z", raw_json={})])
        rows = []
        for i in range(1, 12):
            rows.append(PlayerGameweekRecord(player_id=i, event=1, fixture_id=1, minutes=90, starts=1,
                                             was_home=1, red_cards=0, source="element_summary", raw_json={}))
        # player 1 leaves at 30 having been sent off -> dismissal, not an exit
        rows[0] = PlayerGameweekRecord(player_id=1, event=1, fixture_id=1, minutes=30, starts=1,
                                       was_home=1, red_cards=1, source="element_summary", raw_json={})
        rows.append(PlayerGameweekRecord(player_id=12, event=1, fixture_id=1, minutes=20, starts=0,
                                         was_home=1, red_cards=0, source="element_summary", raw_json={}))
        # The evidence window is bounded by the cutoff below (2026-09-10T12:00:00Z)
        # and the fixture's own kickoff, so the rows must be declared as written
        # inside it.  Leaving this to wall-clock now would put them after the
        # cutoff, where the boundary is right to treat them as unavailable.
        repo.upsert_player_gameweeks(conn, rows, "2026-09-10T08:00:00Z")
    evidence = sm.audit_substitution_evidence(conn, 2, "2026-09-10T12:00:00Z")
    assert evidence["red_card_exits"] == 1
    assert 30.0 not in evidence["exit_minutes"]  # the dismissal is not an exit


def test_p60_start_is_derived_from_event_timing_not_independent():
    side = _side_eleven()
    profile = _profile(side)
    derived = sm.derive_from_profile(side, profile, CFG)
    starters = [d for d in derived if d["p_start"] > 0]
    for row in starters:
        before_60 = sum(m for m, t in zip(row_exit_bands(row, profile), CFG.band_representative_minutes) if t < 60)
        expected = 1.0 - before_60 / row["p_start"]
        assert row["p_60_given_start"] == pytest.approx(max(0.0, min(1.0, expected)), abs=1e-9)


def row_exit_bands(row, profile):
    return profile["players"][row["index"]]["exit_mass_by_band"]


def test_p80_start_is_derived_from_event_timing():
    side = _side_eleven()
    profile = _profile(side)
    derived = sm.derive_from_profile(side, profile, CFG)
    for row in [d for d in derived if d["p_start"] > 0]:
        bands = profile["players"][row["index"]]["exit_mass_by_band"]
        before_80 = sum(m for m, t in zip(bands, CFG.band_representative_minutes) if t < 80)
        assert row["p_80_given_start"] == pytest.approx(1.0 - before_80 / row["p_start"], abs=1e-9)


def test_expected_minutes_if_start_is_derived_from_event_timing():
    side = _side_eleven()
    profile = _profile(side)
    derived = sm.derive_from_profile(side, profile, CFG)
    for row in [d for d in derived if d["p_start"] > 0]:
        bands = profile["players"][row["index"]]["exit_mass_by_band"]
        exit_total = sum(bands)
        expected = ((row["p_start"] - exit_total) * 90.0
                    + sum(m * t for m, t in zip(bands, CFG.band_representative_minutes))) / row["p_start"]
        assert row["expected_minutes_if_start"] == pytest.approx(expected, abs=1e-9)


def test_expected_minutes_if_cameo_is_derived_from_event_timing():
    side = _side_eleven()
    profile = _profile(side)
    derived = sm.derive_from_profile(side, profile, CFG)
    for row in [d for d in derived if d["p_cameo"] > 0]:
        bands = profile["players"][row["index"]]["entry_mass_by_band"]
        expected = sum(m * (90.0 - t) for m, t in zip(bands, CFG.band_representative_minutes)) / row["p_cameo"]
        assert row["expected_minutes_if_cameo"] == pytest.approx(expected, abs=1e-9)


def test_p60_cameo_only_from_sufficiently_early_entries():
    side = _side_eleven()
    profile = _profile(side)
    derived = sm.derive_from_profile(side, profile, CFG)
    for row in derived:
        bands = profile["players"][row["index"]]["entry_mass_by_band"]
        early = sum(m for m, t in zip(bands, CFG.band_representative_minutes) if (90.0 - t) >= 60.0)
        expected = (early / row["p_cameo"]) if row["p_cameo"] > 0 else 0.0
        assert row["p_60_given_cameo"] == pytest.approx(max(0.0, min(1.0, expected)), abs=1e-9)
    # With this band set only the 0-29 band can yield a 60+ cameo.
    assert all(row["p_60_given_cameo"] <= 0.01 for row in derived if row["p_cameo"] > 0)


def test_team_identities_hold():
    identities = sm.derive_team_identities(sm.derive_from_profile(_side_eleven(), _profile(), CFG), CFG)
    assert identities["p_start_total"] == pytest.approx(11.0, abs=1e-5)
    assert identities["exit_mass"] == pytest.approx(identities["entry_mass"], abs=1e-6)
    assert identities["expected_minutes"] == pytest.approx(990.0, abs=1e-3)


def test_gk_and_outfield_start_mass_in_the_real_model(tmp_path):
    """The REAL end-to-end model must produce a coherent positional start mass.

    One goalkeeper and ten outfielders per side, 990 minutes, and matching exit
    and entry mass -- on real model output flowing through the substitution
    layer, not on the pure solver's synthetic payloads.

    Hermetic by construction: a temporary database with its own completed
    history, whose write times are stated relative to the synthetic cutoff.  The
    test previously read the live database at a fixed historical cutoff, so it
    depended on what an official refresh last wrote: the sanctioned refresh
    advanced ``updated_at`` on every historical row, the causal boundary
    correctly refused that cutoff's evidence, and the model raised.  A test that
    a routine refresh can break is measuring the database, not the model.
    """

    from fpl_brain import repositories as repo
    from fpl_brain.database import connect_database
    from fpl_brain.models import PlayerGameweekRecord
    from test_minutes_coherence import CUTOFF, EVENT, HISTORY_OBSERVED_AT, _world

    conn = connect_database(tmp_path / "substitution.db")
    _world(conn)
    # Genuine substitution evidence on the completed fixture: two players who
    # came off the bench, written after kickoff and before the cutoff.  Without
    # a non-starter who played there is no entry mass to be coherent about.
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=25, event=1, fixture_id=90, was_home=1, minutes=20,
                                     starts=0, total_points=1, source="element_summary", raw_json={}),
                PlayerGameweekRecord(player_id=125, event=1, fixture_id=90, was_home=0, minutes=15,
                                     starts=0, total_points=1, source="element_summary", raw_json={}),
            ],
            HISTORY_OBSERVED_AT,
        )

    rows, profiles = sm.build_minutes_predictions_substitution_coherent(conn, EVENT, CUTOFF)

    assert profiles, "the real model produced no side profiles"
    assert all(p["status"] == "COHERENT" for p in profiles)
    for profile in profiles:
        identities = profile["identities"]
        assert identities["p_start_total"] == pytest.approx(11.0, abs=1e-4)
        assert identities["expected_minutes"] == pytest.approx(990.0, abs=1e-2)
        assert identities["exit_mass"] == pytest.approx(identities["entry_mass"], abs=1e-6)

    # The property the test is named for: exactly one goalkeeper and ten
    # outfielders of start mass on every side of the real output.
    team_of = {
        int(row[0]): int(row[1])
        for row in conn.execute("SELECT id, team_id FROM players WHERE team_id IS NOT NULL")
    }
    sides: dict = {}
    for row in rows:
        sides.setdefault((int(row["fixture_id"]), team_of[int(row["player_id"])]), []).append(row)
    assert len(sides) == len(profiles)
    for (fixture_id, team_id), side in sorted(sides.items()):
        keeper = sum(float(r["p_start"]) for r in side if int(r.get("position_id") or 0) == 1)
        outfield = sum(float(r["p_start"]) for r in side if int(r.get("position_id") or 0) != 1)
        assert keeper == pytest.approx(1.0, abs=1e-4), f"fixture {fixture_id} team {team_id} goalkeeper mass"
        assert outfield == pytest.approx(10.0, abs=1e-4), f"fixture {fixture_id} team {team_id} outfield mass"
    conn.close()


def test_impossible_allocation_fails():
    # A side with no start capacity cannot host exit mass.
    side = [{"p_start": 0.0, "p_available": 1.0, "p80_given_start": 0.0,
             "cameo_propensity": 0.0, "expected_minutes_if_start": 0.0} for _ in range(11)]
    profile = _profile(side)
    assert "SUBSTITUTION_MASS_INFEASIBLE" in profile["risk_flags"]


def test_deterministic_output():
    first = _profile(_side_eleven())
    second = _profile(_side_eleven())
    assert first == second


def test_config_hash_changes_with_tunables():
    assert sm.SubstitutionConfig().config_hash() != sm.SubstitutionConfig(team_evidence_prior_strength=6.0).config_hash()
    assert sm.SubstitutionConfig().config_hash() != sm.SubstitutionConfig(gk_substitution_prior=0.01).config_hash()


def test_band_representatives_are_config_driven():
    cfg = sm.SubstitutionConfig()
    assert len(cfg.time_bands) == len(cfg.band_representative_minutes)
    for (low, high), rep in zip(cfg.time_bands, cfg.band_representative_minutes):
        assert low <= rep <= high


def test_mc_consumes_the_same_substitution_profile():
    import sqlite3

    import fpl_brain.monte_carlo as mc

    conn = sqlite3.connect("file:K:/FPL/fpl.db?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    fixtures = mc.load_fixture_inputs(conn, event=4, xpts_run_id=33, minutes_run_id=27, team_run_id=28)
    conn.close()
    # the frozen minutes run 27 predates v1.4, so no profile is attached there;
    # the alignment is exercised by building a side that carries one.
    side = {
        "team_id": 1,
        "players": [
            {"player_id": i, "position": "GKP" if i == 1 else "MID",
             "minutes": {"p_start": 1.0 if i <= 11 else 0.0, "p_cameo": 0.0 if i <= 11 else 0.15,
                         "p_available": 1.0, "p_60_given_start": 0.9, "p_80_given_start": 0.8,
                         "expected_minutes_if_start": 90.0, "expected_minutes_if_cameo": 15.0,
                         "p_60_given_cameo": 0.04},
             "payload": {"fixture_xg_per90": 0.2, "fixture_xa_per90": 0.2, "yellow_per90": 0.0,
                         "defcon_actions_per90": 0.0,
                         "save_model": {"saves_per90_posterior": 0.0, "pressure_multiplier": 1.0}}}
            for i in range(1, 26)
        ],
        "substitution_profile": {"p_sub_count": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                 "event_time_bands": [[0, 29], [30, 44], [45, 59], [60, 74], [75, 89]],
                                 "event_time_masses": [0, 0, 0, 0, 5.0],
                                 "gk_event_mass": 0.0},
    }
    rng = random.Random(1)
    counts = []
    for _ in range(200):
        world = mc._sample_side_world(side, rng, mc.MonteCarloConfig())
        counts.append(world["substitute_entrants"])
    # K is drawn from the profile's distribution (always 5 here)
    assert set(counts) == {5}
    assert all(abs(sum(w["minutes"].values()) - 990.0) < 1e-9
               for w in [mc._sample_side_world(side, random.Random(2), mc.MonteCarloConfig()) for _ in range(20)])


# ---------------------------------------------------------------------------
# PE1-P1-02 — a scheduled placeholder is not substitution evidence
# ---------------------------------------------------------------------------


def test_a_post_kickoff_placeholder_does_not_create_a_false_team_fixture():
    """A placeholder written after kickoff satisfies every clause the old
    composed boundary had, so it reached the side grouping and manufactured a
    team-fixture that contained no substitution evidence at all.  Those are
    not harmless: ``team_fixtures`` is the denominator of
    ``expected_substitutions``, so each phantom side dilutes the league rate.
    """

    from fpl_brain import repositories as repo
    from fpl_brain.database import connect_database
    from fpl_brain.models import (
        EventRecord, FixtureRecord, PlayerGameweekRecord, PlayerRecord, PositionRecord, TeamRecord,
    )

    cutoff = "2026-09-10T12:00:00Z"
    observed_at = "2026-09-10T08:00:00Z"

    def build(with_placeholder_side: bool):
        conn = connect_database(":memory:")
        with conn:
            repo.upsert_teams(conn, [TeamRecord(id=1, name="A"), TeamRecord(id=2, name="B"),
                                     TeamRecord(id=3, name="C"), TeamRecord(id=4, name="D")])
            repo.upsert_positions(conn, [PositionRecord(id=p, singular_name_short=n)
                                         for p, n in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))])
            repo.upsert_players(conn, [PlayerRecord(id=i, web_name=f"P{i}", team_id=1, element_type=3)
                                       for i in range(1, 40)])
            repo.upsert_events(conn, [EventRecord(id=e, finished=1, data_checked=1, raw_json={})
                                      for e in (1, 2)])
            repo.upsert_fixtures(conn, [
                # Fixture 1: a real, fully reported team-1 vs team-2 match.
                FixtureRecord(id=1, event=1, team_h=1, team_a=2, finished=1, started=1,
                              kickoff_time="2026-08-22T14:00:00Z", raw_json={}),
                # Fixture 2: finished and pre-cutoff, but its rows were never refreshed.
                FixtureRecord(id=2, event=1, team_h=3, team_a=4, finished=1, started=1,
                              kickoff_time="2026-08-22T16:00:00Z", raw_json={}),
            ])
            rows = [PlayerGameweekRecord(player_id=i, event=1, fixture_id=1, was_home=1,
                                         minutes=90, starts=1, red_cards=0,
                                         source="element_summary", raw_json={}) for i in range(1, 12)]
            rows += [PlayerGameweekRecord(player_id=i, event=1, fixture_id=1, was_home=0,
                                          minutes=90, starts=1, red_cards=0,
                                          source="element_summary", raw_json={}) for i in range(12, 23)]
            repo.upsert_player_gameweeks(conn, rows, observed_at)
            if with_placeholder_side:
                # Written AFTER fixture 2's kickoff, so only the value signature
                # can reject it: minutes=0 and no performance column at all.
                placeholders = [
                    PlayerGameweekRecord(player_id=i, event=1, fixture_id=2, was_home=side,
                                         minutes=0, source="element_summary", raw_json={})
                    for side, span in ((1, range(23, 34)), (0, range(34, 39)))
                    for i in span
                ]
                repo.upsert_player_gameweeks(conn, placeholders, "2026-08-23T09:00:00Z")
        return conn

    clean = sm.audit_substitution_evidence(build(False), 2, cutoff)
    assert clean["team_fixtures"] == 2

    polluted = sm.audit_substitution_evidence(build(True), 2, cutoff)
    assert polluted["team_fixtures"] == clean["team_fixtures"], (
        "a placeholder-only side manufactured a false zero-substitution team-fixture"
    )
    assert polluted["expected_substitutions"] == clean["expected_substitutions"]
    assert polluted["k_counts"] == clean["k_counts"]
