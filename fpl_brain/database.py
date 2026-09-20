"""SQLite connection setup and additive migrations."""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path
from typing import Callable

from .utils import utc_now

SCHEMA_VERSION = 16

# Model families the analytics spine may record.  ``baseline`` / ``minutes_v1``
# predate this list; the team and player-rate families were added in m007 and
# the deterministic xPts family in m008.
RUN_MODEL_FAMILIES = (
    "baseline",
    "minutes_v1",
    "team_strength_v1",
    "team_baseline",
    "player_rates_v1",
    "player_rate_baseline",
    "xpts_v1",
    "monte_carlo_v1",
)


def m001_initial(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fetch_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          started_at TEXT NOT NULL,
          finished_at TEXT,
          status TEXT NOT NULL CHECK (status IN ('running','success','partial','failed')),
          trigger TEXT NOT NULL,
          current_event INTEGER,
          endpoints_ok TEXT,
          endpoints_failed TEXT,
          error_message TEXT,
          raw_dir TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY,
          name TEXT,
          deadline_time TEXT,
          deadline_time_epoch INTEGER,
          finished INTEGER,
          data_checked INTEGER,
          is_previous INTEGER,
          is_current INTEGER,
          is_next INTEGER,
          average_entry_score INTEGER,
          highest_score INTEGER,
          most_selected INTEGER,
          most_transferred_in INTEGER,
          most_captained INTEGER,
          most_vice_captained INTEGER,
          top_element INTEGER,
          transfers_made INTEGER,
          released INTEGER,
          raw_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chip_definitions (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          number INTEGER,
          chip_type TEXT,
          start_event INTEGER,
          stop_event INTEGER,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS teams (
          id INTEGER PRIMARY KEY,
          code INTEGER,
          name TEXT NOT NULL,
          short_name TEXT,
          strength INTEGER,
          strength_overall_home INTEGER,
          strength_overall_away INTEGER,
          strength_attack_home INTEGER,
          strength_attack_away INTEGER,
          strength_defence_home INTEGER,
          strength_defence_away INTEGER,
          raw_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
          id INTEGER PRIMARY KEY,
          singular_name TEXT,
          singular_name_short TEXT,
          plural_name TEXT,
          squad_select INTEGER,
          squad_min_play INTEGER,
          squad_max_play INTEGER,
          raw_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS players (
          id INTEGER PRIMARY KEY,
          code INTEGER,
          first_name TEXT,
          second_name TEXT,
          web_name TEXT NOT NULL,
          full_name TEXT,
          norm_name TEXT,
          team_id INTEGER REFERENCES teams(id),
          element_type INTEGER REFERENCES positions(id),
          squad_number INTEGER,
          opta_code TEXT,
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          is_active INTEGER NOT NULL DEFAULT 1,
          raw_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS player_snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          player_id INTEGER NOT NULL REFERENCES players(id),
          fetch_run_id INTEGER NOT NULL REFERENCES fetch_runs(id),
          captured_at TEXT NOT NULL,
          event_context INTEGER,
          now_cost INTEGER,
          cost_change_event INTEGER,
          cost_change_start INTEGER,
          selected_by_percent REAL,
          transfers_in_event INTEGER,
          transfers_out_event INTEGER,
          transfers_in INTEGER,
          transfers_out INTEGER,
          status TEXT,
          news TEXT,
          news_added TEXT,
          chance_of_playing_this_round INTEGER,
          chance_of_playing_next_round INTEGER,
          total_points INTEGER,
          event_points INTEGER,
          points_per_game REAL,
          form REAL,
          minutes INTEGER,
          starts INTEGER,
          goals_scored INTEGER,
          assists INTEGER,
          clean_sheets INTEGER,
          goals_conceded INTEGER,
          saves INTEGER,
          bonus INTEGER,
          bps INTEGER,
          yellow_cards INTEGER,
          red_cards INTEGER,
          penalties_saved INTEGER,
          penalties_missed INTEGER,
          influence REAL,
          creativity REAL,
          threat REAL,
          ict_index REAL,
          expected_goals REAL,
          expected_assists REAL,
          expected_goal_involvements REAL,
          expected_goals_conceded REAL,
          expected_goals_per_90 REAL,
          expected_assists_per_90 REAL,
          expected_goal_involvements_per_90 REAL,
          expected_goals_conceded_per_90 REAL,
          defensive_contribution INTEGER,
          clearances_blocks_interceptions INTEGER,
          recoveries INTEGER,
          tackles INTEGER,
          ep_this REAL,
          ep_next REAL,
          value_form REAL,
          value_season REAL,
          raw_json TEXT NOT NULL,
          UNIQUE (player_id, fetch_run_id)
        );

        CREATE TABLE IF NOT EXISTS fixtures (
          id INTEGER PRIMARY KEY,
          code INTEGER,
          event INTEGER,
          kickoff_time TEXT,
          team_h INTEGER REFERENCES teams(id),
          team_a INTEGER REFERENCES teams(id),
          team_h_score INTEGER,
          team_a_score INTEGER,
          team_h_difficulty INTEGER,
          team_a_difficulty INTEGER,
          started INTEGER,
          finished INTEGER,
          finished_provisional INTEGER,
          provisional_start_time INTEGER,
          minutes INTEGER,
          stats_json TEXT,
          raw_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS player_gameweeks (
          player_id INTEGER NOT NULL REFERENCES players(id),
          event INTEGER NOT NULL,
          fixture_id INTEGER,
          opponent_team INTEGER,
          was_home INTEGER,
          kickoff_time TEXT,
          minutes INTEGER,
          starts INTEGER,
          total_points INTEGER,
          goals_scored INTEGER,
          assists INTEGER,
          clean_sheets INTEGER,
          goals_conceded INTEGER,
          saves INTEGER,
          bonus INTEGER,
          bps INTEGER,
          yellow_cards INTEGER,
          red_cards INTEGER,
          penalties_saved INTEGER,
          penalties_missed INTEGER,
          own_goals INTEGER,
          influence REAL,
          creativity REAL,
          threat REAL,
          ict_index REAL,
          expected_goals REAL,
          expected_assists REAL,
          expected_goal_involvements REAL,
          expected_goals_conceded REAL,
          defensive_contribution INTEGER,
          value INTEGER,
          selected INTEGER,
          transfers_in INTEGER,
          transfers_out INTEGER,
          transfers_balance INTEGER,
          source TEXT NOT NULL,
          raw_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (player_id, event, fixture_id)
        );

        CREATE TABLE IF NOT EXISTS manager_state (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          entry_id INTEGER NOT NULL,
          fetch_run_id INTEGER REFERENCES fetch_runs(id),
          captured_at TEXT NOT NULL,
          event INTEGER,
          player_name TEXT,
          team_name TEXT,
          summary_overall_points INTEGER,
          summary_overall_rank INTEGER,
          summary_event_points INTEGER,
          summary_event_rank INTEGER,
          bank INTEGER,
          team_value INTEGER,
          total_transfers INTEGER,
          event_transfers INTEGER,
          event_transfers_cost INTEGER,
          points_on_bench INTEGER,
          active_chip TEXT,
          free_transfers_manual INTEGER,
          raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS manager_chips (
          entry_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          event INTEGER NOT NULL,
          time TEXT,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (entry_id, name, event)
        );

        CREATE TABLE IF NOT EXISTS squad_picks (
          entry_id INTEGER NOT NULL,
          event INTEGER NOT NULL,
          player_id INTEGER NOT NULL REFERENCES players(id),
          position INTEGER NOT NULL,
          multiplier INTEGER,
          is_captain INTEGER,
          is_vice_captain INTEGER,
          is_starting INTEGER,
          synced_at TEXT NOT NULL,
          raw_json TEXT NOT NULL,
          PRIMARY KEY (entry_id, event, player_id)
        );

        CREATE TABLE IF NOT EXISTS watchlist (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          player_id INTEGER NOT NULL REFERENCES players(id),
          status TEXT NOT NULL CHECK (status IN ('WATCH','BUY','HOLD','AVOID','SELL')),
          reason TEXT,
          target_event_from INTEGER,
          target_event_to INTEGER,
          notes TEXT,
          is_active INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scouting_imports (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          source_file TEXT NOT NULL,
          file_sha256 TEXT NOT NULL,
          schema_version TEXT,
          agent TEXT,
          generated_at TEXT,
          gameweek INTEGER,
          players_total INTEGER,
          players_resolved INTEGER,
          notes_inserted INTEGER,
          unresolved_json TEXT,
          imported_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scouting_notes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          import_id INTEGER REFERENCES scouting_imports(id),
          player_id INTEGER NOT NULL REFERENCES players(id),
          key TEXT NOT NULL,
          category TEXT,
          value_text TEXT,
          value_num REAL,
          value_unit TEXT,
          confidence TEXT CHECK (confidence IN ('low','medium','high')),
          evidence_json TEXT,
          observation TEXT,
          gameweek_context INTEGER,
          observed_at TEXT NOT NULL,
          expires_at TEXT,
          agent TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS decisions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event INTEGER NOT NULL,
          action TEXT NOT NULL,
          player_in_id INTEGER REFERENCES players(id),
          player_out_id INTEGER REFERENCES players(id),
          captain_id INTEGER REFERENCES players(id),
          vice_captain_id INTEGER REFERENCES players(id),
          chip TEXT,
          reasoning TEXT,
          confidence TEXT CHECK (confidence IN ('low','medium','high')),
          assumptions_json TEXT,
          invalidators_json TEXT,
          expected_cost INTEGER,
          created_at TEXT NOT NULL,
          review_notes TEXT,
          reviewed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS strategy (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event INTEGER,
          wildcard_horizon TEXT,
          bench_boost_plan TEXT,
          free_hit_plan TEXT,
          triple_captain_plan TEXT,
          risk_posture TEXT,
          notes TEXT,
          created_at TEXT NOT NULL
        );

        CREATE VIEW IF NOT EXISTS v_scouting_current AS
        SELECT s.* FROM (
          SELECT notes.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY notes.player_id, notes.key
                   ORDER BY notes.observed_at DESC, notes.id DESC
                 ) AS current_rank
          FROM scouting_notes notes
        ) s
        WHERE s.current_rank = 1;

        CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
        CREATE INDEX IF NOT EXISTS idx_players_type ON players(element_type);
        CREATE INDEX IF NOT EXISTS idx_players_norm_name ON players(norm_name);
        CREATE INDEX IF NOT EXISTS idx_players_web_name ON players(web_name);
        CREATE INDEX IF NOT EXISTS idx_snap_player_time ON player_snapshots(player_id, captured_at);
        CREATE INDEX IF NOT EXISTS idx_snap_run ON player_snapshots(fetch_run_id);
        CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event);
        CREATE INDEX IF NOT EXISTS idx_fixtures_teams ON fixtures(team_h, team_a);
        CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff ON fixtures(kickoff_time);
        CREATE INDEX IF NOT EXISTS idx_pgw_event ON player_gameweeks(event);
        CREATE INDEX IF NOT EXISTS idx_scouting_player ON scouting_notes(player_id, key, observed_at);
        CREATE INDEX IF NOT EXISTS idx_scouting_expiry ON scouting_notes(expires_at);
        CREATE INDEX IF NOT EXISTS idx_squad_event ON squad_picks(entry_id, event);
        CREATE INDEX IF NOT EXISTS idx_manager_time ON manager_state(entry_id, captured_at);
        CREATE INDEX IF NOT EXISTS idx_watchlist_active ON watchlist(is_active, status);
        CREATE INDEX IF NOT EXISTS idx_decisions_event ON decisions(event);
        """
    )
    now = utc_now()
    conn.execute("INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1')")
    conn.execute("INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('created_at', ?)", (now,))


def m002_manual_manager_state(conn: sqlite3.Connection) -> None:
    """Add event-scoped manager-entered transfer state without changing FACT rows."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS manager_manual_state (
          entry_id INTEGER NOT NULL,
          event INTEGER NOT NULL,
          free_transfers INTEGER,
          bank INTEGER,
          source TEXT NOT NULL DEFAULT 'manual',
          captured_at TEXT NOT NULL,
          PRIMARY KEY (entry_id, event)
        );

        CREATE TABLE IF NOT EXISTS manager_selling_prices (
          entry_id INTEGER NOT NULL,
          event INTEGER NOT NULL,
          player_id INTEGER NOT NULL,
          selling_price INTEGER NOT NULL,
          source TEXT NOT NULL DEFAULT 'manual',
          captured_at TEXT NOT NULL,
          PRIMARY KEY (entry_id, event, player_id)
        );

        CREATE INDEX IF NOT EXISTS idx_manager_manual_state_event
          ON manager_manual_state(entry_id, event);
        CREATE INDEX IF NOT EXISTS idx_manager_selling_prices_event
          ON manager_selling_prices(entry_id, event);
        """
    )


def m003_manager_value_engine(conn: sqlite3.Connection) -> None:
    """Add acquisition history and comparable-market metadata for value resolution."""

    conn.executescript(
        """
        ALTER TABLE manager_selling_prices ADD COLUMN market_price_at_capture INTEGER;

        CREATE TABLE IF NOT EXISTS manager_player_acquisitions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          entry_id INTEGER NOT NULL,
          player_id INTEGER NOT NULL REFERENCES players(id),
          acquired_event INTEGER NOT NULL,
          purchase_price INTEGER NOT NULL,
          sold_event INTEGER,
          source TEXT NOT NULL CHECK (source IN ('verified_initial_squad','official_transfer_history','manual','reconciled')),
          acquired_at TEXT,
          sold_at TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_manager_acquisitions_entry_player
          ON manager_player_acquisitions(entry_id, player_id, acquired_event);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_manager_active_acquisition
          ON manager_player_acquisitions(entry_id, player_id)
          WHERE sold_event IS NULL;
        """
    )


def m004_manager_state_observations(conn: sqlite3.Connection) -> None:
    """Add append-only observation history for manual manager state and prices.

    The current-state tables keep their exact `(entry, event)` scoping so every
    existing reader is unchanged, while each manual entry is additionally
    recorded as an immutable observation so multiple valid states inside one
    event (for example before and after a transfer) are never lost.
    """

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS manager_state_observations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          entry_id INTEGER NOT NULL,
          event INTEGER NOT NULL,
          free_transfers INTEGER,
          bank INTEGER,
          source TEXT NOT NULL DEFAULT 'manual',
          captured_at TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS manager_selling_price_observations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          entry_id INTEGER NOT NULL,
          event INTEGER NOT NULL,
          player_id INTEGER NOT NULL,
          selling_price INTEGER NOT NULL,
          market_price_at_capture INTEGER,
          source TEXT NOT NULL DEFAULT 'manual',
          captured_at TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_manager_state_observations
          ON manager_state_observations(entry_id, event, captured_at, id);
        CREATE INDEX IF NOT EXISTS idx_manager_selling_price_observations
          ON manager_selling_price_observations(entry_id, event, player_id, captured_at, id);
        """
    )
    # Seed the observation history from the existing current-state rows so an
    # upgrade never loses the one state the old overwrite-keyed table held.
    conn.execute(
        """INSERT INTO manager_state_observations(entry_id,event,free_transfers,bank,source,captured_at,created_at)
           SELECT entry_id,event,free_transfers,bank,source,captured_at,captured_at
             FROM manager_manual_state
            WHERE NOT EXISTS (
              SELECT 1 FROM manager_state_observations o
               WHERE o.entry_id=manager_manual_state.entry_id
                 AND o.event=manager_manual_state.event
                 AND o.captured_at=manager_manual_state.captured_at
            )"""
    )
    conn.execute(
        """INSERT INTO manager_selling_price_observations(
             entry_id,event,player_id,selling_price,market_price_at_capture,source,captured_at,created_at)
           SELECT entry_id,event,player_id,selling_price,market_price_at_capture,source,captured_at,captured_at
             FROM manager_selling_prices
            WHERE NOT EXISTS (
              SELECT 1 FROM manager_selling_price_observations o
               WHERE o.entry_id=manager_selling_prices.entry_id
                 AND o.event=manager_selling_prices.event
                 AND o.player_id=manager_selling_prices.player_id
                 AND o.captured_at=manager_selling_prices.captured_at
            )"""
    )


def m005_analytics_spine(conn: sqlite3.Connection) -> None:
    """Prospective-analytics spine: immutable projection freezes and calibration.

    Additive. One ProjectionRun is one immutable analytics generation; every
    frozen Prediction is append-only, protected below at the storage level by
    triggers. Outcome observations and calibration records are append channels
    keyed for idempotency, never overwriting frozen rows.
    """

    conn.executescript(
        """CREATE TABLE IF NOT EXISTS projection_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          model_family TEXT NOT NULL CHECK (model_family IN ('baseline','minutes_v1')),
          model_version TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          planning_event INTEGER NOT NULL,
          planning_context_hash TEXT,
          data_cutoff TEXT NOT NULL,
          scouting_cutoff TEXT,
          official_run_ids TEXT,
          code_revision TEXT,
          config_hash TEXT,
          random_seed INTEGER,
          deadline_status TEXT CHECK (deadline_status IN ('PRE_DEADLINE','LATE_FREEZE')),
          status TEXT NOT NULL CHECK (status IN ('running','complete','failed'))
        );

        CREATE TABLE IF NOT EXISTS frozen_predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          kind TEXT NOT NULL CHECK (kind IN (
            'OFFICIAL_FPL_EP_NEXT','RECENT_POINTS_BASELINE','NAIVE_P90_BASELINE',
            'NAIVE_MINUTES_BASELINE','MINUTES_V1')),
          player_id INTEGER NOT NULL,
          fixture_id INTEGER REFERENCES fixtures(id),
          event INTEGER NOT NULL,
          payload_json TEXT NOT NULL,
          model_version TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          UNIQUE (projection_run_id, kind, player_id, fixture_id, event)
        );

        CREATE TABLE IF NOT EXISTS outcome_observations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event INTEGER NOT NULL,
          player_id INTEGER NOT NULL,
          fixture_id INTEGER,
          actual_started INTEGER,
          actual_minutes INTEGER,
          actual_60_plus INTEGER,
          actual_zero_minutes INTEGER,
          actual_points INTEGER,
          actual_xg REAL,
          actual_xa REAL,
          actual_defcon INTEGER,
          observed_at TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'player_gameweeks_final',
          UNIQUE (event, player_id, fixture_id)
        );

        CREATE TABLE IF NOT EXISTS calibration_records (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          event INTEGER NOT NULL,
          metric_name TEXT NOT NULL,
          sample_count INTEGER NOT NULL,
          metric_value REAL NOT NULL,
          model_version TEXT,
          generated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_projection_runs_event
          ON projection_runs(planning_event, model_family);
        CREATE INDEX IF NOT EXISTS idx_frozen_predictions_run
          ON frozen_predictions(projection_run_id, kind);
        CREATE INDEX IF NOT EXISTS idx_frozen_predictions_event
          ON frozen_predictions(event, kind);
        CREATE INDEX IF NOT EXISTS idx_outcomes_event
          ON outcome_observations(event);
        CREATE INDEX IF NOT EXISTS idx_calibration_run
          ON calibration_records(projection_run_id, event, metric_name);

        CREATE TRIGGER IF NOT EXISTS frozen_predictions_no_update
          BEFORE UPDATE ON frozen_predictions
          BEGIN SELECT RAISE(ABORT, 'frozen predictions are immutable'); END;

        CREATE TRIGGER IF NOT EXISTS frozen_predictions_no_delete
          BEFORE DELETE ON frozen_predictions
          BEGIN SELECT RAISE(ABORT, 'frozen predictions are immutable'); END;
        """
    )



def m006_prior_season_and_run_immutability(conn: sqlite3.Connection) -> None:
    """Add the previous-season prior store and lock completed projection runs.

    `player_season_histories` stores official element-summary `history_past`
    rows (proven fields: season_name, starts, minutes, total_points; no
    official appearances field).  Completed projection runs become immutable
    at the storage level; only a `running` run can still be finished.
    """

    conn.executescript(
        """CREATE TABLE IF NOT EXISTS player_season_histories (
          player_id INTEGER NOT NULL REFERENCES players(id),
          season_name TEXT NOT NULL,
          minutes INTEGER,
          starts INTEGER,
          total_points INTEGER,
          goals_scored INTEGER,
          assists INTEGER,
          clean_sheets INTEGER,
          bonus INTEGER,
          saves INTEGER,
          source TEXT NOT NULL DEFAULT "element_summary_history_past",
          observed_at TEXT NOT NULL,
          raw_json TEXT NOT NULL,
          UNIQUE (player_id, season_name)
        );
        CREATE INDEX IF NOT EXISTS idx_player_season_histories
          ON player_season_histories(player_id, season_name);

        CREATE TRIGGER IF NOT EXISTS projection_runs_completed_immutable
          BEFORE UPDATE ON projection_runs
          WHEN OLD.status = 'complete'
          BEGIN SELECT RAISE(ABORT, 'completed projection runs are immutable'); END;

        CREATE TRIGGER IF NOT EXISTS projection_runs_no_delete
          BEFORE DELETE ON projection_runs
          BEGIN SELECT RAISE(ABORT, 'projection runs are provenance and cannot be deleted'); END;
        """
    )


def m007_team_and_player_rate_projections(conn: sqlite3.Connection) -> None:
    """Team-strength and player-rate projection stores (analytics Phase 2+3).

    Widens the ``projection_runs.model_family`` CHECK to admit the new run
    families and adds two append-only projection tables that hang off the
    existing run spine (same provenance/immutability semantics, no parallel
    versioning system).

    SQLite cannot ALTER a CHECK constraint, and rebuilding ``projection_runs``
    would require detaching the ``frozen_predictions`` foreign key.  The CHECK
    is therefore widened with the documented ``writable_schema`` text edit:
    it changes only the constraint text, moves no rows, and leaves every
    foreign key untouched.  The edit is atomic (it runs inside the migration
    transaction) and is a no-op if the constraint is already widened.
    """

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='projection_runs'"
    ).fetchone()
    if row is None:
        raise sqlite3.OperationalError("m007: projection_runs table is missing")
    create_sql = row[0]
    old_check = "CHECK (model_family IN ('baseline','minutes_v1'))"
    new_check = (
        "CHECK (model_family IN ('baseline','minutes_v1','team_strength_v1',"
        "'team_baseline','player_rates_v1','player_rate_baseline'))"
    )
    if old_check in create_sql:
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql=? WHERE type='table' AND name='projection_runs'",
            (create_sql.replace(old_check, new_check),),
        )
        conn.execute("PRAGMA writable_schema=OFF")
        # Force SQLite to re-read the schema with the widened constraint.
        conn.execute(
            "PRAGMA schema_version = %d"
            % (int(conn.execute("PRAGMA schema_version").fetchone()[0]) + 1)
        )
    elif new_check not in create_sql:
        raise sqlite3.OperationalError(
            "m007: unexpected projection_runs model_family CHECK; refusing an unsafe edit"
        )

    conn.executescript(
        """CREATE TABLE IF NOT EXISTS team_fixture_projections (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
          event INTEGER NOT NULL,
          team_id INTEGER NOT NULL,
          opponent_id INTEGER NOT NULL,
          venue TEXT NOT NULL CHECK (venue IN ('home','away')),
          payload_json TEXT NOT NULL,
          model_version TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          UNIQUE (projection_run_id, fixture_id, team_id)
        );

        CREATE TABLE IF NOT EXISTS player_rate_projections (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          player_id INTEGER NOT NULL,
          component TEXT NOT NULL,
          event INTEGER NOT NULL,
          payload_json TEXT NOT NULL,
          model_version TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          UNIQUE (projection_run_id, player_id, component)
        );

        CREATE INDEX IF NOT EXISTS idx_team_fixture_projections_run
          ON team_fixture_projections(projection_run_id, event);
        CREATE INDEX IF NOT EXISTS idx_team_fixture_projections_fixture
          ON team_fixture_projections(fixture_id, team_id);
        CREATE INDEX IF NOT EXISTS idx_player_rate_projections_run
          ON player_rate_projections(projection_run_id, component);

        CREATE TRIGGER IF NOT EXISTS team_fixture_projections_no_update
          BEFORE UPDATE ON team_fixture_projections
          BEGIN SELECT RAISE(ABORT, 'team fixture projections are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS team_fixture_projections_no_delete
          BEFORE DELETE ON team_fixture_projections
          BEGIN SELECT RAISE(ABORT, 'team fixture projections are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS player_rate_projections_no_update
          BEFORE UPDATE ON player_rate_projections
          BEGIN SELECT RAISE(ABORT, 'player rate projections are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS player_rate_projections_no_delete
          BEFORE DELETE ON player_rate_projections
          BEGIN SELECT RAISE(ABORT, 'player rate projections are immutable'); END;
        """
    )


def m008_xpts_projections(conn: sqlite3.Connection) -> None:
    """Deterministic expected-FPL-points projections (Analytics Phase 4).

    Widens the ``projection_runs.model_family`` CHECK to admit ``xpts_v1`` and
    adds one append-only player-fixture xPts table on the existing run spine.
    The CHECK is widened with the same contained ``writable_schema`` text edit
    used by m007 (no row movement, no foreign-key change).
    """

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='projection_runs'"
    ).fetchone()
    if row is None:
        raise sqlite3.OperationalError("m008: projection_runs table is missing")
    create_sql = row[0]
    old_check = (
        "CHECK (model_family IN ('baseline','minutes_v1','team_strength_v1',"
        "'team_baseline','player_rates_v1','player_rate_baseline'))"
    )
    new_check = (
        "CHECK (model_family IN ('baseline','minutes_v1','team_strength_v1',"
        "'team_baseline','player_rates_v1','player_rate_baseline','xpts_v1'))"
    )
    if old_check in create_sql:
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql=? WHERE type='table' AND name='projection_runs'",
            (create_sql.replace(old_check, new_check),),
        )
        conn.execute("PRAGMA writable_schema=OFF")
        conn.execute(
            "PRAGMA schema_version = %d"
            % (int(conn.execute("PRAGMA schema_version").fetchone()[0]) + 1)
        )
    elif new_check not in create_sql:
        raise sqlite3.OperationalError(
            "m008: unexpected projection_runs model_family CHECK; refusing an unsafe edit"
        )

    conn.executescript(
        """CREATE TABLE IF NOT EXISTS player_fixture_xpts_projections (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          player_id INTEGER NOT NULL,
          fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
          event INTEGER NOT NULL,
          team_id INTEGER NOT NULL,
          opponent_id INTEGER NOT NULL,
          position TEXT NOT NULL,
          minutes_run_id INTEGER NOT NULL,
          team_run_id INTEGER NOT NULL,
          rate_run_id INTEGER NOT NULL,
          payload_json TEXT NOT NULL,
          model_version TEXT NOT NULL,
          scoring_rules_version TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          UNIQUE (projection_run_id, player_id, fixture_id)
        );

        CREATE INDEX IF NOT EXISTS idx_player_fixture_xpts_run
          ON player_fixture_xpts_projections(projection_run_id, event);
        CREATE INDEX IF NOT EXISTS idx_player_fixture_xpts_player
          ON player_fixture_xpts_projections(player_id, fixture_id);

        CREATE TRIGGER IF NOT EXISTS player_fixture_xpts_no_update
          BEFORE UPDATE ON player_fixture_xpts_projections
          BEGIN SELECT RAISE(ABORT, 'xPts projections are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS player_fixture_xpts_no_delete
          BEFORE DELETE ON player_fixture_xpts_projections
          BEGIN SELECT RAISE(ABORT, 'xPts projections are immutable'); END;
        """
    )


def m009_team_minutes_coherence(conn: sqlite3.Connection) -> None:
    """Compact first-class team-coherence records (Phase 4 acceptance pass).

    One append-only row per (projection run, fixture, team) recording the raw
    versus coherent lineup-mass sums, the shared solver intercepts, and the
    constraint residuals.  No run-family CHECK change is needed: the coherent
    challenger stays in the ``minutes_v1`` family with a new model version.
    """

    conn.executescript(
        """CREATE TABLE IF NOT EXISTS team_minutes_coherence (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
          team_id INTEGER NOT NULL,
          players INTEGER NOT NULL,
          raw_start_sum REAL NOT NULL,
          adjusted_start_sum REAL NOT NULL,
          raw_minutes_sum REAL NOT NULL,
          adjusted_minutes_sum REAL NOT NULL,
          start_intercept REAL NOT NULL,
          cameo_intercept REAL NOT NULL,
          start_residual REAL NOT NULL,
          minutes_residual REAL NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE (projection_run_id, fixture_id, team_id)
        );

        CREATE INDEX IF NOT EXISTS idx_team_minutes_coherence_run
          ON team_minutes_coherence(projection_run_id);

        CREATE TRIGGER IF NOT EXISTS team_minutes_coherence_no_update
          BEFORE UPDATE ON team_minutes_coherence
          BEGIN SELECT RAISE(ABORT, 'team coherence records are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS team_minutes_coherence_no_delete
          BEFORE DELETE ON team_minutes_coherence
          BEGIN SELECT RAISE(ABORT, 'team coherence records are immutable'); END;
        """
    )


def m010_monte_carlo_distributions(conn: sqlite3.Connection) -> None:
    """Monte Carlo distribution summaries (Analytics Phase 5).

    Widens the ``projection_runs.model_family`` CHECK to admit
    ``monte_carlo_v1`` and adds one append-only table of per-player-fixture
    distribution summaries (means, quantiles, tail probabilities and
    reconciliation diagnostics) -- never the raw draws.
    """

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='projection_runs'"
    ).fetchone()
    if row is None:
        raise sqlite3.OperationalError("m010: projection_runs table is missing")
    create_sql = row[0]
    old_check = (
        "CHECK (model_family IN ('baseline','minutes_v1','team_strength_v1',"
        "'team_baseline','player_rates_v1','player_rate_baseline','xpts_v1'))"
    )
    new_check = (
        "CHECK (model_family IN ('baseline','minutes_v1','team_strength_v1',"
        "'team_baseline','player_rates_v1','player_rate_baseline','xpts_v1','monte_carlo_v1'))"
    )
    if old_check in create_sql:
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql=? WHERE type='table' AND name='projection_runs'",
            (create_sql.replace(old_check, new_check),),
        )
        conn.execute("PRAGMA writable_schema=OFF")
        conn.execute(
            "PRAGMA schema_version = %d"
            % (int(conn.execute("PRAGMA schema_version").fetchone()[0]) + 1)
        )
    elif new_check not in create_sql:
        raise sqlite3.OperationalError(
            "m010: unexpected projection_runs model_family CHECK; refusing an unsafe edit"
        )

    conn.executescript(
        """CREATE TABLE IF NOT EXISTS monte_carlo_distributions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          player_id INTEGER NOT NULL,
          fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
          event INTEGER NOT NULL,
          team_id INTEGER NOT NULL,
          opponent_id INTEGER NOT NULL,
          position TEXT NOT NULL,
          xpts_run_id INTEGER NOT NULL,
          minutes_run_id INTEGER NOT NULL,
          team_run_id INTEGER NOT NULL,
          rate_run_id INTEGER NOT NULL,
          payload_json TEXT NOT NULL,
          model_version TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          UNIQUE (projection_run_id, player_id, fixture_id)
        );

        CREATE INDEX IF NOT EXISTS idx_monte_carlo_distributions_run
          ON monte_carlo_distributions(projection_run_id, event);

        CREATE TRIGGER IF NOT EXISTS monte_carlo_distributions_no_update
          BEFORE UPDATE ON monte_carlo_distributions
          BEGIN SELECT RAISE(ABORT, 'monte carlo distributions are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS monte_carlo_distributions_no_delete
          BEFORE DELETE ON monte_carlo_distributions
          BEGIN SELECT RAISE(ABORT, 'monte carlo distributions are immutable'); END;
        """
    )


def m011_team_substitution_profiles(conn: sqlite3.Connection) -> None:
    """Compact, append-only team substitution profiles (minutes_v1.4.0).

    One row per (projection run, fixture, team) holding the substitution-count
    distribution and the expected event mass by time band that both the
    deterministic Minutes v1.4 model and the Monte Carlo simulator consume.
    """

    conn.executescript(
        """CREATE TABLE IF NOT EXISTS team_substitution_profiles (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
          team_id INTEGER NOT NULL,
          expected_substitutions REAL NOT NULL,
          expected_exit_mass REAL NOT NULL,
          expected_entry_mass REAL NOT NULL,
          gk_event_mass REAL,
          evidence_matches INTEGER,
          prior_source TEXT,
          status TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE (projection_run_id, fixture_id, team_id)
        );

        CREATE INDEX IF NOT EXISTS idx_team_substitution_profiles_run
          ON team_substitution_profiles(projection_run_id);

        CREATE TRIGGER IF NOT EXISTS team_substitution_profiles_no_update
          BEFORE UPDATE ON team_substitution_profiles
          BEGIN SELECT RAISE(ABORT, 'team substitution profiles are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS team_substitution_profiles_no_delete
          BEFORE DELETE ON team_substitution_profiles
          BEGIN SELECT RAISE(ABORT, 'team substitution profiles are immutable'); END;
        """
    )


def m012_projection_run_source_snapshot(conn: sqlite3.Connection) -> None:
    """Record the deterministic source-snapshot hash on every projection run.

    Provenance only: the worktree is intentionally dirty/uncommitted, so a
    reproducible hash over the exact relevant source files (path + bytes, sorted)
    ties a certification artifact to the code that produced it.
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(projection_runs)")}
    if "source_snapshot_sha256" not in columns:
        conn.execute("ALTER TABLE projection_runs ADD COLUMN source_snapshot_sha256 TEXT")


def m013_event_start_free_transfers(conn: sqlite3.Connection) -> None:
    """Explicit provenance for the FT bank at the START of the Gameweek.

    A Wildcard (or Free Hit) played after transfers were already made must
    preserve the FT state from the start of the Gameweek, not the current
    remaining count.  That value cannot be inferred unambiguously from
    ``free_transfers`` alone, so it is stored explicitly on the manual state
    (current row + append-only observations).  Additive nullable columns only:
    no rebuild, no backfill, and NULL means "not stated" rather than a guess.
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(manager_manual_state)")}
    if "event_start_free_transfers" not in columns:
        conn.execute("ALTER TABLE manager_manual_state ADD COLUMN event_start_free_transfers INTEGER")
    observation_columns = {row[1] for row in conn.execute("PRAGMA table_info(manager_state_observations)")}
    if "event_start_free_transfers" not in observation_columns:
        conn.execute("ALTER TABLE manager_state_observations ADD COLUMN event_start_free_transfers INTEGER")


def m014_execution_control(conn: sqlite3.Connection) -> None:
    """Operational execution provenance for the production run controller (R2A).

    This is **operational** provenance, deliberately separate from the
    predictive ``projection_runs`` spine: it records who ran what, under which
    lease, against which deadline, and how the run ended.  It is purely additive
    (four new tables and their indexes); no existing table, row, constraint or
    foreign key is touched, so append-only projection immutability is unaffected.

    ``execution_leases`` carries a partial unique index over ``status='ACTIVE'``
    so at most one ACTIVE lease can exist per ``(lease_kind, lease_key)``.  That
    is the database-level enforcement of single-run and single-writer semantics:
    it does not depend on a lock file existing, and releasing or reclaiming a
    lease frees the key immediately.
    """

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS execution_runs (
          run_uuid TEXT PRIMARY KEY,
          label TEXT,
          planning_event INTEGER,
          planning_cutoff TEXT,
          semantic_run_key TEXT NOT NULL,
          status TEXT NOT NULL CHECK (status IN (
            'CREATED','RUNNING','CANCEL_REQUESTED','CANCELLED','FAILED','COMPLETE')),
          current_stage TEXT,
          owner_pid INTEGER,
          owner_host TEXT,
          started_at TEXT,
          hard_stop_at TEXT,
          heartbeat_at TEXT,
          cancel_requested_at TEXT,
          cancel_reason TEXT,
          finished_at TEXT,
          failure_reason TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_execution_runs_status
          ON execution_runs(status);
        CREATE INDEX IF NOT EXISTS idx_execution_runs_semantic
          ON execution_runs(semantic_run_key);

        CREATE TABLE IF NOT EXISTS execution_leases (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lease_kind TEXT NOT NULL CHECK (lease_kind IN ('run','writer')),
          lease_key TEXT NOT NULL,
          run_uuid TEXT NOT NULL,
          owner_pid INTEGER NOT NULL,
          owner_host TEXT,
          acquired_at TEXT NOT NULL,
          heartbeat_at TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          released_at TEXT,
          status TEXT NOT NULL CHECK (status IN ('ACTIVE','RELEASED','EXPIRED','RECLAIMED')),
          reclaims_lease_id INTEGER,
          reclaimed_by_lease_id INTEGER,
          note TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_leases_active
          ON execution_leases(lease_kind, lease_key) WHERE status = 'ACTIVE';
        CREATE INDEX IF NOT EXISTS idx_execution_leases_run
          ON execution_leases(run_uuid);

        CREATE TABLE IF NOT EXISTS execution_children (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_uuid TEXT NOT NULL REFERENCES execution_runs(run_uuid),
          pid INTEGER NOT NULL,
          argv_json TEXT NOT NULL,
          registered_at TEXT NOT NULL,
          terminated_at TEXT,
          exit_status TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_execution_children_run
          ON execution_children(run_uuid);

        CREATE TABLE IF NOT EXISTS execution_stages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_uuid TEXT NOT NULL REFERENCES execution_runs(run_uuid),
          stage TEXT NOT NULL,
          semantic_key TEXT NOT NULL,
          status TEXT NOT NULL CHECK (status IN (
            'NOT_STARTED','RUNNING','COMPLETE','FAILED','CANCELLED','SKIPPED_INSUFFICIENT_TIME')),
          restart_class TEXT,
          started_at TEXT,
          finished_at TEXT,
          detail_json TEXT,
          created_at TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_stages_complete
          ON execution_stages(semantic_key, stage) WHERE status = 'COMPLETE';
        CREATE INDEX IF NOT EXISTS idx_execution_stages_run
          ON execution_stages(run_uuid);
        """
    )


def m015_bootstrap_generation_identity(conn: sqlite3.Connection) -> None:
    """Certified official-player-pool identity (R4B.1.1).

    The identity of the latest ACCEPTED official bootstrap generation, persisted
    INSIDE SQLite so that it is automatically carried into every execution
    snapshot (R4A.3 snapshots this database with ``VACUUM INTO``).  A JSON side
    report cannot be the causal authority for a decision, because the decision's
    causal source is the immutable snapshot.

    One row per bootstrap ingest attempt.  ``accepted`` distinguishes a
    generation that was allowed to redefine the official pool from one that was
    rejected for implausible drift: a rejected generation is recorded for audit
    but must never be read as authoritative, which the reader enforces with
    ``WHERE accepted=1``.

    ``element_ids_json`` stores the exact sorted official id set so the pool can
    be verified by IDENTITY, not merely by count.  ``element_ids_sha256`` is the
    deterministic digest of that same sorted set.

    Purely additive: one new table and its index; no existing table, row,
    constraint, foreign key or ``projection_runs`` row is touched.
    """

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS bootstrap_generations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fetch_run_id INTEGER,
          captured_at TEXT NOT NULL,
          accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
          official_element_count INTEGER NOT NULL,
          parsed_count INTEGER NOT NULL,
          persisted_count INTEGER NOT NULL,
          element_ids_sha256 TEXT NOT NULL,
          element_ids_json TEXT NOT NULL,
          availability_counts_json TEXT,
          club_player_counts_json TEXT,
          acceptance_rule TEXT NOT NULL,
          rejection_reasons_json TEXT,
          acceptance_rule_version TEXT NOT NULL,
          recorded_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_bootstrap_generations_accepted
          ON bootstrap_generations(accepted, id);
        CREATE INDEX IF NOT EXISTS idx_bootstrap_generations_captured
          ON bootstrap_generations(captured_at);
        """
    )


def m016_outcome_ledger(conn: sqlite3.Connection) -> None:
    """PE-5 append-only outcome history and generation-certified freeze provenance.

    ``player_gameweeks`` is CURRENT / LATEST-STATE storage: the same logical
    player/event/fixture row is refreshed in place, so what an earlier read
    observed is silently replaced.  ``outcome_observations`` (m005) is keyed
    ``UNIQUE(event, player_id, fixture_id)`` and upserts in place for the same
    reason -- it answers "what is the outcome now", never "what did we know
    then".  PE-5 needs the second question answered, so this migration adds the
    durable point-in-time history rather than a second current-state table.

    Three additive tables, no row movement and no existing table, constraint or
    foreign key touched:

    * ``outcome_observation_captures`` -- append-only observation history.  A
      capture is immutable at the storage level (UPDATE and DELETE both abort),
      so a correction is necessarily a NEW capture naming what it supersedes.
      ``capture_digest`` is the idempotency key: re-ingesting the SAME
      observation (same identity, same source identity, same capture time, same
      payload) is a no-op, while the same observation captured at a different
      time is a second retained row.  ``event_time``, ``official_final_at`` and
      ``captured_at`` are three separate columns because the football event, the
      official finalisation and the repository's read are three different
      moments that must not be collapsed into one timestamp.
    * ``prediction_freeze_provenance`` -- the generation certificate for one
      prediction freeze.  ``bootstrap_generation_id`` pins the accepted official
      generation AT CERTIFICATION TIME: a later accepted generation does not
      retroactively certify an earlier freeze, so the recorded id is never
      re-resolved from "the newest accepted row".  It is authoritative only when
      ``provenance_state = 'GENERATION_CERTIFIED'``; a ``NOT_CERTIFIABLE`` record
      keeps the generation it examined, plus the reasons it was refused.
    * ``prediction_freeze_runs`` -- the exact projection runs belonging to that
      freeze, with the partial-unique index making "one run, one freeze" a
      database fact rather than a convention.

    Nothing here is destructive and nothing is backfilled: a pre-PE-5 run simply
    has no provenance row, which reads as the truthful not-fully-certified
    state rather than a manufactured certificate.
    """

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS outcome_observation_captures (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          capture_digest TEXT NOT NULL UNIQUE,
          grain TEXT NOT NULL CHECK (grain IN ('player_event','player_fixture')),
          event INTEGER NOT NULL,
          player_id INTEGER NOT NULL,
          fixture_id INTEGER,
          team_id INTEGER,
          opponent_team_id INTEGER,
          event_time TEXT,
          official_final_at TEXT,
          captured_at TEXT NOT NULL,
          observation_state TEXT NOT NULL CHECK (observation_state IN ('FINAL','PROVISIONAL')),
          supersedes_capture_id INTEGER REFERENCES outcome_observation_captures(id),
          correction_reason TEXT,
          source_name TEXT NOT NULL,
          source_identity TEXT,
          source_payload_sha256 TEXT,
          fetch_run_id INTEGER,
          archive_capture_id TEXT,
          backfill_evidence_json TEXT,
          payload_json TEXT NOT NULL,
          minutes INTEGER,
          starts INTEGER,
          total_points INTEGER,
          goals_scored INTEGER,
          assists INTEGER,
          clean_sheets INTEGER,
          goals_conceded INTEGER,
          saves INTEGER,
          bonus INTEGER,
          bps INTEGER,
          yellow_cards INTEGER,
          red_cards INTEGER,
          penalties_saved INTEGER,
          penalties_missed INTEGER,
          own_goals INTEGER,
          defensive_contribution INTEGER,
          created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_outcome_captures_key
          ON outcome_observation_captures(grain, event, player_id, fixture_id, captured_at, id);
        CREATE INDEX IF NOT EXISTS idx_outcome_captures_event
          ON outcome_observation_captures(event, captured_at);
        CREATE INDEX IF NOT EXISTS idx_outcome_captures_captured
          ON outcome_observation_captures(captured_at);

        CREATE TRIGGER IF NOT EXISTS outcome_observation_captures_no_update
          BEFORE UPDATE ON outcome_observation_captures
          BEGIN SELECT RAISE(ABORT, 'outcome observation captures are append-only history'); END;
        CREATE TRIGGER IF NOT EXISTS outcome_observation_captures_no_delete
          BEFORE DELETE ON outcome_observation_captures
          BEGIN SELECT RAISE(ABORT, 'outcome observation captures are append-only history'); END;

        CREATE TABLE IF NOT EXISTS prediction_freeze_provenance (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          freeze_identity TEXT NOT NULL UNIQUE,
          provenance_state TEXT NOT NULL CHECK (provenance_state IN (
            'GENERATION_CERTIFIED','LEGACY_PROVENANCE','NOT_CERTIFIABLE')),
          planning_event INTEGER NOT NULL,
          planning_cutoff TEXT NOT NULL,
          model_version TEXT,
          config_hash TEXT,
          random_seed INTEGER,
          source_snapshot_sha256 TEXT,
          code_revision TEXT,
          official_fetch_run_id INTEGER,
          bootstrap_generation_id INTEGER,
          bootstrap_captured_at TEXT,
          bootstrap_element_count INTEGER,
          bootstrap_element_ids_sha256 TEXT,
          bootstrap_element_ids_json TEXT,
          bootstrap_acceptance_rule_version TEXT,
          runs_json TEXT NOT NULL,
          prediction_artifact_identity TEXT NOT NULL,
          prediction_artifact_sha256 TEXT NOT NULL,
          reasons_json TEXT NOT NULL,
          certification_version TEXT NOT NULL,
          certified_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_freeze_provenance_event
          ON prediction_freeze_provenance(planning_event, provenance_state);
        CREATE INDEX IF NOT EXISTS idx_freeze_provenance_generation
          ON prediction_freeze_provenance(bootstrap_generation_id);

        CREATE TRIGGER IF NOT EXISTS prediction_freeze_provenance_no_update
          BEFORE UPDATE ON prediction_freeze_provenance
          BEGIN SELECT RAISE(ABORT, 'prediction freeze provenance is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS prediction_freeze_provenance_no_delete
          BEFORE DELETE ON prediction_freeze_provenance
          BEGIN SELECT RAISE(ABORT, 'prediction freeze provenance is immutable'); END;

        CREATE TABLE IF NOT EXISTS prediction_freeze_runs (
          freeze_id INTEGER NOT NULL REFERENCES prediction_freeze_provenance(id),
          projection_run_id INTEGER NOT NULL REFERENCES projection_runs(id),
          model_family TEXT NOT NULL,
          model_version TEXT NOT NULL,
          config_hash TEXT,
          random_seed INTEGER,
          PRIMARY KEY (freeze_id, projection_run_id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_prediction_freeze_runs_run
          ON prediction_freeze_runs(projection_run_id);

        CREATE TRIGGER IF NOT EXISTS prediction_freeze_runs_no_update
          BEFORE UPDATE ON prediction_freeze_runs
          BEGIN SELECT RAISE(ABORT, 'prediction freeze run membership is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS prediction_freeze_runs_no_delete
          BEFORE DELETE ON prediction_freeze_runs
          BEGIN SELECT RAISE(ABORT, 'prediction freeze run membership is immutable'); END;
        """
    )


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    m001_initial,
    m002_manual_manager_state,
    m003_manager_value_engine,
    m004_manager_state_observations,
    m005_analytics_spine,
    m006_prior_season_and_run_immutability,
    m007_team_and_player_rate_projections,
    m008_xpts_projections,
    m009_team_minutes_coherence,
    m010_monte_carlo_distributions,
    m011_team_substitution_profiles,
    m012_projection_run_source_snapshot,
    m013_event_start_free_transfers,
    m014_execution_control,
    m015_bootstrap_generation_identity,
    m016_outcome_ledger,
]


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    current = int(row[0]) if row else 0
    for version, migration in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        with conn:
            migration(conn)
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(version),),
            )


def connect_database(path: str | Path, *, busy_timeout_ms: int = 30000) -> sqlite3.Connection:
    """Open (and migrate) the database as a potential writer.

    ``busy_timeout`` is set so a transient lock waits instead of crashing the
    caller after the driver's default 5s.  It is a resilience measure only: the
    concurrency architecture is the writer lease in ``fpl_brain.execution``, not
    a longer timeout.
    """

    database_path = str(path)
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path, timeout=max(0.0, float(busy_timeout_ms) / 1000.0))
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    initialize_database(conn)
    # Additive compatibility view for databases created before the
    # deterministic v_scouting_current definition was introduced.  No fact
    # rows are changed, and repository reads use this ranked view consistently.
    conn.execute(
        """CREATE VIEW IF NOT EXISTS v_scouting_current_ranked AS
           SELECT s.* FROM (
             SELECT notes.*,
                    ROW_NUMBER() OVER (
                      PARTITION BY notes.player_id, notes.key
                      ORDER BY notes.observed_at DESC, notes.id DESC
                    ) AS current_rank
             FROM scouting_notes notes
           ) s
           WHERE s.current_rank = 1"""
    )
    return conn


# Small compatibility aliases keep the public library surface unsurprising.
get_connection = connect_database
init_db = initialize_database


def connect_readonly_database(
    path: str | Path, *, busy_timeout_ms: int = 30000
) -> sqlite3.Connection:
    """Open an existing database strictly read-only, with no DDL.

    Read-only workers (report builders, forensics, the decision board) must
    coexist with a single writer without widening the write-lock window.  This
    factory never runs migrations and never issues ``CREATE``/``ALTER``, which
    is what made ordinary reads contend with the writer in the R1 audit.
    """

    from pathlib import Path as _Path

    database_path = str(path)
    if database_path != ":memory:" and not _Path(database_path).exists():
        raise FileNotFoundError(f"database not found: {database_path}")
    uri = f"file:{database_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=max(0.0, float(busy_timeout_ms) / 1000.0))
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    conn.execute("PRAGMA query_only=ON")
    return conn


@contextlib.contextmanager
def write_transaction(
    conn: sqlite3.Connection,
    *,
    attempts: int = 5,
    base_delay: float = 0.05,
    sleep: Callable[[float], None] | None = None,
):
    """Explicit ``BEGIN IMMEDIATE`` write transaction with bounded retry.

    Deferred transactions upgrade the lock late, which is the classic
    ``SQLITE_BUSY`` path.  Taking the write lock up front makes contention
    deterministic, and the retry is bounded so a genuine deadlock still fails
    loudly instead of hanging or silently continuing.
    """

    _sleep = sleep or time.sleep
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            last_error = exc
            if attempt + 1 >= max(1, int(attempts)):
                raise
            _sleep(base_delay * (2**attempt))
            continue
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
        return
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error
