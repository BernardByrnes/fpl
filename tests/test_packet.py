"""Canonical decision packet: deterministic rendering and provenance rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]) ) if False else None

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.models import PickRecord, PlayerRecord, PlayerSnapshotRecord, TeamRecord
from fpl_brain.packet import (
    build_decision_packet,
    packet_to_json,
    render_packet_markdown,
    verify_packet_markdown,
)

PLAYER_IDS = list(range(1, 16))


def _config(tmp_path, database: Path):
    return {
        "fpl_entry_id": 241392,
        "season": "2026/27",
        "paths": {"database": str(database), "raw_dir": str(tmp_path / "raw"), "exports_dir": str(tmp_path / "exports")},
        "report": {"scouting_stale_after_days": 14},
    }


def _seed(conn):
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="Arsenal U", short_name="ARSU")])
        repo.upsert_players(conn, [PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}", team_id=1) for pid in PLAYER_IDS])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=pid, captured_at="2026-08-30T09:00:00Z", now_cost=50, raw_json={}) for pid in PLAYER_IDS],
            run,
        )
        repo.upsert_squad_picks(
            conn,
            241392,
            2,
            [PickRecord(player_id=pid, position=index + 1, raw_json={}) for index, pid in enumerate(PLAYER_IDS)],
        )
        for pid in PLAYER_IDS:
            repo.insert_manager_acquisition(
                conn, 241392, pid, 2, 50, source="official_transfer_history", created_at="2026-08-30T10:00:00Z"
            )
        repo.upsert_manual_manager_state(conn, 241392, 2, 3, 0, captured_at="2026-08-29T12:00:00Z")


def test_packet_metadata_and_canonical_sections(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    _seed(conn)
    built = build_decision_packet(conn, _config(tmp_path, database), 2, as_of="2026-08-30T12:00:00Z")
    conn.close()
    packet = built["packet"]
    metadata = packet["metadata"]
    assert metadata["entry_id"] == 241392
    assert metadata["season"] == "2026/27"
    assert metadata["event"] == 2
    assert metadata["as_of"] == "2026-08-30T12:00:00Z"
    assert "health" in metadata and "official_runs" in metadata
    context = packet["planning_context"]
    assert context["manager_state"]["free_transfers"] == 3
    assert len(packet["owned_players"]) == 15
    # Canonical numbers: purchase/market/effective sell trace back to canonical data.
    first = packet["owned_players"][0]
    sell = first["selling_price"]
    assert sell["purchase_price"] == 50 and sell["official_market_price"] == 50
    assert sell["effective_selling_price"] == 50


def test_json_to_markdown_determinism(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    _seed(conn)
    built = build_decision_packet(conn, _config(tmp_path, database), 2, as_of="2026-08-30T12:00:00Z")
    conn.close()
    packet = built["packet"]
    markdown = render_packet_markdown(packet)
    assert render_packet_markdown(packet) == markdown
    # The stored JSON is the source artifact: rendering the reloaded JSON
    # object must reproduce the stored Markdown byte for byte.
    reloaded = json.loads(packet_to_json(packet))
    assert render_packet_markdown(reloaded) == markdown
    assert verify_packet_markdown(markdown, reloaded) is True

    # A manually restated number disagrees with the canonical packet and must
    # be detected (tamper detection).
    md_lines = markdown.splitlines()
    tampered = md_lines.copy()
    for index, line in enumerate(tampered):
        if "purchase £5.0m" in line:
            tampered[index] = line.replace("purchase £5.0m", "purchase £9.9m")
            break
    else:
        raise AssertionError("expected a rendered purchase price line")
    assert verify_packet_markdown("\n".join(tampered) + "\n", reloaded) is False


def test_packet_rendering_contains_health_gaps(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    _seed(conn)
    with conn:
        # Remove one official market snapshot to force a selling DATA GAP FAIL.
        conn.execute("DELETE FROM player_snapshots WHERE player_id=1")
    built = build_decision_packet(conn, _config(tmp_path, database), 2, as_of="2026-08-30T12:00:00Z")
    conn.close()
    markdown = render_packet_markdown(built["packet"])
    assert built["context"].health["status"] in {"FAIL", "WARN"}
    assert "Data health:" in markdown
    assert "## Data Gaps" in markdown
