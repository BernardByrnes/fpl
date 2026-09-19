"""PE-3 Step 1 — the optional Monte Carlo raw-event capture (`capture_bps_worlds`).

The challenger needs per player x fixture x WORLD raw football events.  The simulator
already samples them, but the RAW forms are discarded after scoring: the save COUNT is
reduced to ``floor(saves / saves_per_point)`` points, and ``conceded_on_pitch`` never
leaves ``_score_personal_events``.  This side-channel observes those exact values.

Contract pinned here:
  * OPT-IN: nothing is allocated unless requested, and the returned dict differs only by
    the new field;
  * it consumes ZERO randomness and adds no RNG stream, so every historical output is
    bit-identical with the capture on or off;
  * it captures RAW EVENT COUNTS (not FPL points) for the WHOLE fixture, including
    zero-minute players, at player x fixture x world grain.
"""

from __future__ import annotations

from fpl_brain import monte_carlo as mc
from test_monte_carlo import RULES, _fixture, _finish

FIXTURE_ID = 100
WORLDS = 40
SEED = 12345


def _run(*, capture: bool, worlds: int = WORLDS, seed: int = SEED, fixture=None):
    fixture = _finish(fixture or _fixture())
    config = mc.MonteCarloConfig(simulations=worlds, seed=seed)
    return fixture, mc.simulate({fixture["fixture_id"]: fixture}, config, RULES,
                                capture_bps_worlds=capture), config


def _strip(result: dict) -> dict:
    """Everything except the new field — the exact comparison the brief specifies."""

    return {key: value for key, value in result.items() if key != "bps_worlds"}


# ---------------------------------------------------------------------------
# API AND GRAIN
# ---------------------------------------------------------------------------


def test_the_capture_is_opt_in_and_absent_by_default():
    _fixture_obj, off, _config = _run(capture=False)
    assert "bps_worlds" in off
    assert off["bps_worlds"] is None


def test_the_capture_is_keyed_by_fixture_then_world():
    fixture, on, config = _run(capture=True)
    captured = on["bps_worlds"]
    assert set(captured) == {fixture["fixture_id"]}
    assert len(captured[fixture["fixture_id"]]) == config.simulations
    for world in captured[fixture["fixture_id"]]:
        assert isinstance(world, dict) and world


def test_grain_is_player_x_fixture_x_world_not_gameweek():
    fixture, on, _config = _run(capture=True, worlds=5)
    for world in on["bps_worlds"][fixture["fixture_id"]]:
        for record in world.values():
            assert set(record) == {"position", "minutes", "goals_scored", "assists",
                                   "clean_sheets", "goals_conceded", "saves",
                                   "yellow_cards"}


def test_the_whole_fixture_is_captured_not_just_captured_player_ids():
    """Bonus is a whole-fixture competition, so every simulated player must be present."""

    fixture, on, _config = _run(capture=True, worlds=3)
    world = on["bps_worlds"][fixture["fixture_id"]][0]
    expected = {int(player["player_id"])
                for side in fixture["sides"] for player in side["players"]}
    assert set(world) == expected
    assert len(expected) > 15, "a fixture has far more than one manager's squad"


def test_positions_are_captured_per_world():
    fixture, on, _config = _run(capture=True, worlds=2)
    world = on["bps_worlds"][fixture["fixture_id"]][0]
    expected = {int(player["player_id"]): player["position"]
                for side in fixture["sides"] for player in side["players"]}
    for player_id, record in world.items():
        assert record["position"] == expected[player_id]
        assert record["position"] in {"GKP", "DEF", "MID", "FWD"}


def test_zero_minute_players_appear_with_explicit_zero_counts():
    fixture, on, _config = _run(capture=True, worlds=WORLDS)
    zero_minute = [(world, pid, record)
                   for world in on["bps_worlds"][fixture["fixture_id"]]
                   for pid, record in world.items() if record["minutes"] == 0.0]
    assert zero_minute, "the synthetic fixture has non-playing squad members"
    for _world, _pid, record in zero_minute:
        assert record["minutes"] == 0.0
        for field in ("goals_scored", "assists", "clean_sheets", "goals_conceded",
                      "saves", "yellow_cards"):
            assert record[field] == 0, field


# ---------------------------------------------------------------------------
# RAW VALUES, NOT POINT DEDUCTIONS
# ---------------------------------------------------------------------------


def test_gk_saves_are_the_raw_count_not_the_floor_divided_points():
    """Raw save COUNT, not the points the count is worth and not floor(seen / threshold).

    The shared fixture models keepers who never save, so this builds its own fixture with a
    real save rate; otherwise the assertion would pass vacuously on an all-zero capture.
    """

    built = _fixture()
    # `_side` does not forward overrides to its two keepers, so set the keepers' save
    # model directly; otherwise every capture is a vacuous zero.
    for side in built["sides"]:
        for player in side["players"]:
            if player["position"] == "GKP":
                player["payload"]["save_model"] = {"saves_per90_posterior": 6.0,
                                                   "pressure_multiplier": 1.0}
    fixture, on, _config = _run(capture=True, worlds=400, fixture=built)
    counts = [record["saves"]
              for world in on["bps_worlds"][fixture["fixture_id"]]
              for record in world.values() if record["position"] == "GKP"]
    assert counts, "expected at least one goalkeeper"
    assert all(isinstance(value, int) and value >= 0 for value in counts)
    assert max(counts) >= 3, f"expected a multi-save world, saw {sorted(set(counts))}"

    # The captured values must be the SAVE COUNT, so they are unbounded by the divisor:
    # the points path collapses the same draws with floor(saves / saves_per_point).
    divisor = RULES.saves_per_point
    assert divisor > 1, "this test only discriminates while the divisor exceeds one"
    assert any(value > 1 for value in counts), (
        f"raw counts must exceed the divisor's plateau, saw {sorted(set(counts))}"
    )


def test_goals_conceded_is_the_on_pitch_count_not_the_points_deduction():
    """The capture must carry the RAW count; the point value floors it per 2 conceded."""

    fixture, on, _config = _run(capture=True, worlds=WORLDS)
    counts = {record["goals_conceded"]
              for world in on["bps_worlds"][fixture["fixture_id"]]
              for record in world.values()}
    assert any(value > 2 for value in counts), (
        f"the point deduction would cap information; raw counts seen: {sorted(counts)}"
    )


def test_non_goalkeepers_carry_explicit_zero_saves():
    fixture, on, _config = _run(capture=True, worlds=5)
    for world in on["bps_worlds"][fixture["fixture_id"]]:
        for record in world.values():
            if record["position"] != "GKP":
                assert record["saves"] == 0


def test_yellow_is_the_realised_event_flag():
    fixture, on, _config = _run(capture=True, worlds=200)
    values = {record["yellow_cards"]
              for world in on["bps_worlds"][fixture["fixture_id"]]
              for record in world.values()}
    assert values <= {0, 1}
    assert values, "expected at least one captured player"


def test_goals_and_assists_are_counts_not_flags():
    """A multi-goal world must report 2, not a 0/1 flag."""

    fixture, on, _config = _run(capture=True, worlds=400)
    goals = {record["goals_scored"]
             for world in on["bps_worlds"][fixture["fixture_id"]]
             for record in world.values()}
    assert any(value >= 1 for value in goals), f"expected scorers, saw {sorted(goals)}"
    for world in on["bps_worlds"][fixture["fixture_id"]]:
        for record in world.values():
            assert record["clean_sheets"] in (0, 1)


def test_captured_goal_counts_match_the_production_summaries():
    """The capture must observe the SAME goals production scored, not a second draw."""

    fixture, on, _config = _run(capture=True, worlds=200)
    per_player_capture = {}
    for world in on["bps_worlds"][fixture["fixture_id"]]:
        for player_id, record in world.items():
            per_player_capture[player_id] = per_player_capture.get(player_id, 0) + record["goals_scored"]

    for summary in on["summaries"]:
        player_id = int(summary["player_id"])
        production_goals = round(float(summary["mean_goal_count"]) * summary["simulations"])
        assert per_player_capture.get(player_id, 0) == production_goals, player_id


def test_minutes_are_the_physical_sampled_minutes():
    fixture, on, _config = _run(capture=True, worlds=WORLDS)
    for summary in on["summaries"]:
        player_id = int(summary["player_id"])
        captured = [world[player_id]["minutes"]
                    for world in on["bps_worlds"][fixture["fixture_id"]]]
        assert sum(captured) / len(captured) == summary["mean_minutes"]


# ---------------------------------------------------------------------------
# INVARIANCE — the hard gate
# ---------------------------------------------------------------------------


PINNED_FIELDS = ("summaries", "team_minutes", "invalid_lineups", "excess_substitutes",
                 "minute_mass_violations", "occupancy_violations", "occupancy_examples",
                 "suppressed_gk_cameo_mass_mean", "primitive_reconstruction",
                 "calibration", "config_hash", "world_matrix")


def test_capture_off_and_on_are_identical_outside_the_new_field():
    _fixture_obj, off, _config = _run(capture=False)
    _fixture_obj, on, _config = _run(capture=True)
    assert _strip(off) == _strip(on)


def test_every_pinned_output_field_is_identical():
    _fixture_obj, off, _config = _run(capture=False)
    _fixture_obj, on, _config = _run(capture=True)
    for field in PINNED_FIELDS:
        assert off[field] == on[field], field


def test_production_goal_reconciliation_is_unchanged():
    """The frozen zero-variance / production-expectation path must be untouched."""

    _fixture_obj, off, _config = _run(capture=False)
    _fixture_obj, on, _config = _run(capture=True)
    for left, right in zip(off["summaries"], on["summaries"]):
        assert left["mean_reconciliation_error"] == right["mean_reconciliation_error"]
        assert left["standardised_error"] == right["standardised_error"]
        assert left["zero_variance_mismatch"] == right["zero_variance_mismatch"]
        assert left["zero_variance_below_materiality"] == right["zero_variance_below_materiality"]


def test_both_captures_can_run_together_without_disturbing_either():
    fixture = _finish(_fixture())
    config = mc.MonteCarloConfig(simulations=WORLDS, seed=SEED)
    ids = [int(player["player_id"]) for side in fixture["sides"] for player in side["players"]][:20]

    worlds_only = mc.simulate({fixture["fixture_id"]: fixture}, config, RULES,
                              capture_player_ids=ids)
    bps_only = mc.simulate({fixture["fixture_id"]: fixture}, config, RULES,
                           capture_bps_worlds=True)
    both = mc.simulate({fixture["fixture_id"]: fixture}, config, RULES,
                       capture_player_ids=ids, capture_bps_worlds=True)

    assert both["world_matrix"] == worlds_only["world_matrix"]
    assert both["bps_worlds"] == bps_only["bps_worlds"]
    assert _strip(both) == _strip(worlds_only)


def test_the_capture_consumes_no_randomness():
    """Source-level pin: the capture path constructs no RNG."""

    import inspect

    source = inspect.getsource(mc.simulate)
    capture_block = source.split("if capture_bps_worlds:")[1:]
    assert capture_block, "expected a capture block"
    for block in capture_block:
        head = block[:600]
        for banned in ("_stream(", "_poisson(", "random("):
            assert banned not in head, f"capture block must not draw: {banned}"


def test_model_version_is_unchanged():
    assert mc.MONTE_CARLO_MODEL_VERSION == "mc_v1.3.0"


def test_no_config_field_was_added_for_the_capture():
    from dataclasses import fields

    names = {field_.name for field_ in fields(mc.MonteCarloConfig)}
    assert "capture_bps_worlds" not in names
    assert not any("bps" in name for name in names)
