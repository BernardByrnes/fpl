from __future__ import annotations

from fpl_brain.parsers import (
    parse_bootstrap,
    parse_element_summary,
    parse_entry,
    parse_entry_history,
    parse_event_live,
    parse_snapshot,
)


def test_live_samples_parse_and_preserve_uncertain_fields(fixture_json):
    payload = fixture_json("bootstrap_static_sample.json")
    parsed = parse_bootstrap(payload, "2026-08-19T14:58:00Z")
    assert len(parsed.players) == 8
    assert len(parsed.teams) == 3
    assert [position.id for position in parsed.positions] == [1, 2, 3, 4]
    assert parsed.players[0].opta_code
    assert parsed.snapshots[0].selected_by_percent == 34.7
    assert parsed.snapshots[0].expected_goals is not None
    assert parsed.snapshots[0].clearances_blocks_interceptions is not None
    assert parsed.snapshots[0].defensive_contribution is not None

    deleted = dict(payload["elements"][0])
    deleted.pop("selected_by_percent")
    deleted["form"] = "3.25"
    snapshot = parse_snapshot(deleted, "2026-08-19T14:58:00Z")
    assert snapshot is not None
    assert snapshot.selected_by_percent is None
    assert snapshot.form == 3.25
    assert snapshot.chance_of_playing_this_round is None

    extra = dict(payload["elements"][0])
    extra["new_future_field"] = {"kept": True}
    player = parsed.players[0]
    assert player.raw_json["id"] == player.id
    assert parse_snapshot(extra, "2026-08-19T14:58:00Z").raw_json["new_future_field"] == {"kept": True}


def test_summary_history_and_empty_live_are_benign(fixture_json):
    summary = fixture_json("element_summary_sample.json")
    rows = parse_element_summary(summary, 1)
    assert len(rows) == 4
    assert all(row.source == "element_summary" for row in rows)
    assert rows[0].fixture_id is not None
    assert rows[0].event is not None

    assert parse_event_live(fixture_json("event_live_empty.json"), 1) == []
    entry = parse_entry(fixture_json("entry_sample.json"))
    assert entry is not None
    assert entry.team_name == fixture_json("entry_sample.json")["name"]
    nameless_identity = parse_entry({"id": 2, "name": "Only The Team Name"})
    assert nameless_identity is not None and nameless_identity.team_name == "Only The Team Name" and nameless_identity.player_name is None
    assert entry.summary_overall_rank is None
    history = parse_entry_history(fixture_json("entry_history_sample.json"))
    assert history.current == []
    assert history.past[0].overall_rank is not None


def test_event_live_double_gameweek_splits_explain_rows_without_event_total_duplication():
    payload = {
        "elements": [
            {
                "id": 10,
                "stats": {"minutes": 180, "total_points": 14},
                "explain": [
                    {
                        "fixture": 101,
                        "was_home": 1,
                        "opponent_team": 2,
                        "stats": [
                            {"identifier": "minutes", "value": 90, "points": 2},
                            {"identifier": "goals_scored", "value": 1, "points": 5},
                        ],
                    },
                    {
                        "fixture": 102,
                        "was_home": 0,
                        "opponent_team": 3,
                        "stats": [
                            {"identifier": "minutes", "value": 90, "points": 2},
                            {"identifier": "assists", "value": 1, "points": 3},
                        ],
                    },
                ],
            }
        ]
    }
    rows = parse_event_live(payload, 7)
    assert [row.fixture_id for row in rows] == [101, 102]
    assert [row.minutes for row in rows] == [90, 90]
    assert [row.total_points for row in rows] == [7, 5]
    assert [row.opponent_team for row in rows] == [2, 3]
    assert [row.was_home for row in rows] == [1, 0]
