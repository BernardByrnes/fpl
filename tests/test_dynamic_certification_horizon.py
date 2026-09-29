from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import certify_four_gw as certifier  # noqa: E402


@pytest.mark.parametrize(
    ("planning_event", "expected"),
    [
        (5, (5, 6, 7, 8)),
        (6, (6, 7, 8, 9)),
        (7, (7, 8, 9, 10)),
        (19, (19, 20, 21, 22)),
    ],
)
def test_canonical_certification_horizon_rolls_with_planning_event(planning_event, expected):
    assert certifier.canonical_certification_events(
        planning_event, last_event=38
    ) == expected


@pytest.mark.parametrize(
    "supplied",
    [
        "6,7,8,10",       # gap in the horizon
        "5,6,7,8",        # wrong starting event
        "6,7,8,9,10",     # five events
        "6,8,9,10",       # non-contiguous and wrong length
        "6,7,7,9",        # duplicate event
        "",               # empty string
        (),               # empty sequence
    ],
)
def test_noncanonical_event_lists_are_rejected(supplied):
    with pytest.raises(ValueError):
        certifier.canonical_certification_events(
            6, last_event=38, supplied_events=supplied
        )


def test_explicit_canonical_event_lists_are_accepted():
    assert certifier.canonical_certification_events(
        6, last_event=38, supplied_events="6,7,8,9"
    ) == (6, 7, 8, 9)


@pytest.mark.parametrize(
    ("planning_event", "last_event", "expected"),
    [
        (37, 38, (37, 38)),
        (38, 38, (38,)),
    ],
)
def test_legitimate_season_end_short_horizon_uses_existing_decision_rule(
    planning_event, last_event, expected
):
    assert certifier.canonical_certification_events(
        planning_event, last_event=last_event, supplied_events=expected
    ) == expected


def test_planning_event_defaults_to_unique_official_next_event():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, is_next INTEGER NOT NULL)")
    conn.executemany(
        "INSERT INTO events(id, is_next) VALUES (?, ?)",
        [(5, 0), (6, 1), (7, 0)],
    )
    assert certifier.resolve_planning_event(conn) == 6
    assert certifier.resolve_planning_event(conn, 6) == 6
    with pytest.raises(ValueError, match="disagrees with official is_next"):
        certifier.resolve_planning_event(conn, 5)
    conn.close()


def test_planning_event_resolution_fails_closed_without_unique_official_next_event():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, is_next INTEGER NOT NULL)")
    conn.executemany("INSERT INTO events(id, is_next) VALUES (?, ?)", [(6, 0), (7, 0)])
    with pytest.raises(ValueError, match="no unique is_next"):
        certifier.resolve_planning_event(conn)
    conn.executemany("INSERT INTO events(id, is_next) VALUES (?, ?)", [(8, 1), (9, 1)])
    with pytest.raises(ValueError, match="multiple next events"):
        certifier.resolve_planning_event(conn)
    conn.close()
