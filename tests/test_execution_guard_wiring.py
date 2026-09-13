"""R2B: top-level execution-guard wiring for the two remaining production scripts.

These tests never invoke a real predictive stage.  They point the scripts at a
temporary, empty database through a temporary config, so the run fails fast
(empty planning context) while still proving the guard is acquired *before* any
predictive work and that a duplicate invocation is refused at the lease.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fpl_brain import execution
from fpl_brain.database import connect_database

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import final_operational_refresh_gw04 as refresh  # noqa: E402
import run_four_gw_decision as four_gw  # noqa: E402

CUTOFF = "2026-09-12T10:40:04Z"
PROJECTION_TABLES = (
    "projection_runs",
    "frozen_predictions",
    "player_rate_projections",
    "monte_carlo_distributions",
)


@pytest.fixture
def temp_env(tmp_path):
    """A temporary config + empty database, isolated from the live store."""

    db_path = tmp_path / "fpl.db"
    connect_database(db_path).close()
    config = {
        "fpl_entry_id": 241392,
        "season": "2026/27",
        "paths": {
            "database": str(db_path),
            "raw_dir": str(tmp_path / "raw"),
            "exports_dir": str(tmp_path / "exports"),
        },
        "report": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "raw").mkdir(exist_ok=True)
    return {"config": str(config_path), "db": db_path, "tmp": tmp_path}


def _execution_runs(db_path):
    conn = connect_database(db_path)
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM execution_runs ORDER BY rowid")]
    finally:
        conn.close()


def _projection_count(db_path):
    conn = connect_database(db_path)
    try:
        return sum(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in PROJECTION_TABLES
        )
    finally:
        conn.close()


def test_four_gw_decision_acquires_guard_before_any_predictive_work(temp_env, capsys):
    try:
        code = four_gw.main(["--stage", "bundle", "--event", "4", "--cutoff", CUTOFF, "--config", temp_env["config"]])
    except Exception:
        # An empty store may make the planning context raise rather than refuse.
        code = None
    out = capsys.readouterr().out

    assert "execution guard: run_uuid=" in out
    runs = _execution_runs(temp_env["db"])
    assert len(runs) == 1
    # The guard closed out the run: no execution state is left mid-flight.
    assert runs[0]["status"] in execution.TERMINAL_RUN_STATUSES
    assert runs[0]["status"] != execution.RUN_CREATED
    if code is not None:
        assert code != 0
    # No predictive provenance was created by a refused run.
    assert _projection_count(temp_env["db"]) == 0


def test_four_gw_decision_is_refused_when_the_lease_is_held(temp_env):
    conn = connect_database(temp_env["db"])
    try:
        with execution.event_run_guard(
            conn,
            planning_event=4,
            cutoff=CUTOFF,
            label="holder",
            families=["four_gw_decision"],
        ):
            with pytest.raises(execution.LeaseError):
                four_gw.main(["--stage", "bundle", "--event", "4", "--cutoff", CUTOFF, "--config", temp_env["config"]])
    finally:
        conn.close()
    assert _projection_count(temp_env["db"]) == 0


def test_refresh_acquires_guard_before_any_predictive_work(temp_env, capsys):
    try:
        code = refresh.main(["--config", temp_env["config"], "--cutoff", CUTOFF])
    except Exception:
        code = None
    out = capsys.readouterr().out

    assert "execution guard: run_uuid=" in out
    runs = _execution_runs(temp_env["db"])
    assert len(runs) == 1
    assert runs[0]["status"] in execution.TERMINAL_RUN_STATUSES
    if code is not None:
        assert code != 0
    assert _projection_count(temp_env["db"]) == 0


def test_refresh_is_refused_when_the_lease_is_held(temp_env):
    conn = connect_database(temp_env["db"])
    try:
        with execution.event_run_guard(
            conn,
            planning_event=int(refresh.EVENT),
            cutoff=CUTOFF,
            label="holder",
            families=["operational_refresh"],
        ):
            with pytest.raises(execution.LeaseError):
                refresh.main(["--config", temp_env["config"], "--cutoff", CUTOFF])
    finally:
        conn.close()
    assert _projection_count(temp_env["db"]) == 0


def test_both_scripts_route_through_a_single_guard_entry_point():
    """Structural check: no second, unguarded write path into the controller."""

    for module, families in ((four_gw, "four_gw_decision"), (refresh, "operational_refresh")):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert source.count("execution.event_run_guard(") == 1
        assert f'families=["{families}"]' in source
        assert "entered_guard.__exit__(*sys.exc_info())" in source
        assert "_refuse_execution(guard," in source


def test_refusals_are_recorded_as_failed_runs(temp_env):
    """A deliberate refusal must not be reported as a successful execution."""

    conn = connect_database(temp_env["db"])
    try:
        with execution.event_run_guard(
            conn,
            planning_event=4,
            cutoff=CUTOFF,
            label="refusal-check",
            families=["four_gw_decision"],
        ) as guard:
            run_uuid = guard.run_uuid
            code = four_gw._refuse_execution(guard, 4, "decision horizon incomplete")
            assert code == 4
        row = conn.execute(
            "SELECT status, failure_reason FROM execution_runs WHERE run_uuid=?", (run_uuid,)
        ).fetchone()
        assert row["status"] == execution.RUN_FAILED
        assert row["failure_reason"] == "decision horizon incomplete"
        # The same helper exists in the refresh script.
        assert refresh._refuse_execution is not None
    finally:
        conn.close()
