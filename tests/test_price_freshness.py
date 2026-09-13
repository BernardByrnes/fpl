"""Official-price freshness metadata: a stale official price run must never be
silently presented as current market data."""

from __future__ import annotations

import json

from fpl_brain.database import connect_database
from fpl_brain.packet import build_decision_packet, render_packet_markdown
from fpl_brain.planning import (
    OFFICIAL_PRICE_CURRENT,
    OFFICIAL_PRICE_STALE,
    OFFICIAL_PRICE_UNKNOWN,
    get_planning_context,
    official_price_freshness,
)

from test_planning import _seed


def _config(database, tmp_path, threshold_hours=48):
    return {
        "fpl_entry_id": 241392,
        "season": "2026/27",
        "paths": {"database": str(database), "raw_dir": str(tmp_path / "raw"), "exports_dir": str(tmp_path / "exports")},
        "report": {"scouting_stale_after_days": 14, "official_price_stale_after_hours": threshold_hours},
    }


def test_stale_official_price_run_is_flagged_never_current(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    _seed(conn, eventfinished=True, data_checked=True)
    # Official price run exists at 2026-08-30T09:00:00Z; a later as_of far
    # beyond the configured threshold must be STALE, surfaced in metadata,
    # warnings, health, and the packet — not silently shown as current.
    context = get_planning_context(
        conn, 241392, 2, as_of="2026-09-04T09:00:00Z", official_price_stale_after_hours=48
    )
    freshness = context.official_price_freshness
    assert freshness["freshness"] == OFFICIAL_PRICE_STALE
    assert freshness["latest_official_price_run_at"] == "2026-08-30T09:00:00Z"
    assert freshness["official_price_run_age_hours"] > 48
    assert any("official price data is not verifiably current" in reason for reason in context.health["warn_reasons"])
    # The stored official price is untouched by the flag (no fabrication).
    assert conn.execute("SELECT now_cost FROM player_snapshots WHERE player_id=1").fetchone()[0] == 50

    built = build_decision_packet(conn, _config(database, tmp_path), 2, as_of="2026-09-04T09:00:00Z")
    conn.close()
    packet = built["packet"]
    assert packet["metadata"]["official_price_freshness"] == OFFICIAL_PRICE_STALE
    assert packet["metadata"]["latest_official_price_run_at"] == "2026-08-30T09:00:00Z"
    markdown = render_packet_markdown(packet)
    assert "Official price freshness: STALE" in markdown
    assert "2026-08-30T09:00:00Z" in markdown
    # JSON round-trip keeps the exact same deterministic rendering.
    reloaded = json.loads(json.dumps(packet, sort_keys=True))
    assert render_packet_markdown(reloaded) == markdown


def test_fresh_official_price_run_stays_current(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    _seed(conn, eventfinished=True, data_checked=True)
    context = get_planning_context(
        conn, 241392, 2, as_of="2026-08-30T12:00:00Z", official_price_stale_after_hours=48
    )
    assert context.official_price_freshness["freshness"] == OFFICIAL_PRICE_CURRENT
    assert not any(
        "official price data is not verifiably current" in reason for reason in context.health["warn_reasons"]
    )
    conn.close()


def test_configured_threshold_controls_freshness_boundary(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    _seed(conn, eventfinished=True, data_checked=True)
    wide = get_planning_context(
        conn, 241392, 2, as_of="2026-09-04T09:00:00Z", official_price_stale_after_hours=20000
    )
    assert wide.official_price_freshness["freshness"] == OFFICIAL_PRICE_CURRENT
    tight = get_planning_context(
        conn, 241392, 2, as_of="2026-08-30T12:00:00Z", official_price_stale_after_hours=2
    )
    assert tight.official_price_freshness["freshness"] == OFFICIAL_PRICE_STALE
    # No snapshots at all is honestly UNKNOWN, not silently CURRENT.
    empty = connect_database(tmp_path / "empty.db")
    assert official_price_freshness(empty)["freshness"] == OFFICIAL_PRICE_UNKNOWN
    empty.close()
    conn.close()
