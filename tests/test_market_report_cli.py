from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.market_report import open_read_only_database
from fpl_brain.models import EventRecord, FixtureRecord


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_market_report.py"


def _write_config(tmp_path: Path, database: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "fpl_entry_id": None,
                "paths": {
                    "database": str(database),
                    "raw_dir": str(tmp_path / "raw"),
                    "exports_dir": str(tmp_path / "exports"),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_cli(config_path: Path, output_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--out-dir",
            str(output_root),
            *args,
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_market_report_read_only_connection_rejects_logical_database_writes(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    conn.close()

    read_conn = open_read_only_database(database)
    try:
        with pytest.raises(sqlite3.OperationalError):
            read_conn.execute("CREATE TABLE prohibited_market_report_write (id INTEGER)")
    finally:
        read_conn.close()


def test_market_report_cli_rejects_incomplete_gw_and_allows_stdout_preview_without_writes(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    with conn:
        repo.upsert_events(conn, [EventRecord(id=1, name="Gameweek 1", finished=0, data_checked=0, is_next=1)])
        repo.upsert_fixtures(conn, [FixtureRecord(id=101, event=1, finished=0, raw_json={})])
    conn.close()
    config_path = _write_config(tmp_path, database)
    output_root = tmp_path / "reports"
    before = database.read_bytes()

    normal = _run_cli(config_path, output_root, "--gw", "1")
    assert normal.returncode == 4
    assert "incomplete" in normal.stderr.lower()
    assert not output_root.exists()
    assert database.read_bytes() == before

    preview = _run_cli(
        config_path,
        output_root,
        "--gw",
        "1",
        "--dry-run",
        "--stdout",
        "json",
        "--allow-incomplete",
    )
    assert preview.returncode == 0
    rendered = json.loads(preview.stdout)
    assert rendered["gameweek"]["status"] == "preview_incomplete"
    assert any("PREVIEW ONLY" in gap for gap in rendered["data_gaps"])
    assert not output_root.exists()
    assert database.read_bytes() == before


def test_market_report_cli_dry_run_is_read_only_and_normal_mode_writes_paired_artifacts(tmp_path):
    database = tmp_path / "fpl.db"
    conn = connect_database(database)
    with conn:
        repo.upsert_events(conn, [EventRecord(id=1, name="Gameweek 1", finished=1, data_checked=1, is_previous=1)])
        repo.upsert_fixtures(conn, [FixtureRecord(id=101, event=1, finished=1, raw_json={})])
    conn.close()
    config_path = _write_config(tmp_path, database)
    output_root = tmp_path / "reports"
    before = database.read_bytes()

    dry_run = _run_cli(config_path, output_root, "--gw", "1", "--dry-run")
    assert dry_run.returncode == 0
    assert "dry-run success" in dry_run.stdout
    assert not output_root.exists()
    assert database.read_bytes() == before

    normal = _run_cli(config_path, output_root, "--gw", "1")
    assert normal.returncode == 0
    report_directory = output_root / "gw01"
    assert (report_directory / "post_gw_market_report.json").is_file()
    assert (report_directory / "post_gw_market_report.md").is_file()
    assert database.read_bytes() == before
