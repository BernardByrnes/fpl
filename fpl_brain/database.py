"""SQLite connection setup and additive migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

from .utils import utc_now

SCHEMA_VERSION = 1


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


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [m001_initial]


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


def connect_database(path: str | Path) -> sqlite3.Connection:
    database_path = str(path)
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
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
