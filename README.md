# FPL Brain

FPL Brain is a local, provenance-aware evidence pack for the 2026/27 Fantasy Premier League season. It stores official FPL facts separately from scouting context and human strategy, then renders a Markdown and JSON report for an LLM or manager to reason over. It deliberately does not calculate a composite player score.

## Install and first run

Use Python 3.11+ (the project is also tested here on Python 3.14), install the small dependency set, and copy the example configuration:

```bash
python -m pip install -r requirements.txt
cp config.example.json config.json
python scripts/fetch_fpl.py
python scripts/build_report.py
```

On Windows, use `Copy-Item config.example.json config.json` in place of `cp`. The first fetch creates `fpl.db`, `data/raw/`, and the schema automatically. A normal fetch stores `bootstrap-static/` and `fixtures/`, upserts reference data, and appends one immutable player snapshot per active player. Run it again to create another snapshot generation even when nothing changed.

All timestamps stored by the application are UTC ISO-8601 strings. FPL prices remain integer tenths in SQLite (`80` means £8.0m) and are converted only in reports.

## Configuration

`config.json` is intentionally gitignored. The important settings are:

- `fpl_entry_id`: `null` is valid and keeps the whole fetch/report workflow usable before a Team ID is configured.
- `paths`: database, raw-response, and report locations.
- `manual_overrides.free_transfers`: current free transfers are not exposed by the public API; leave null or enter the value manually. Reports label it `[MANUAL]`.
- `strategy`: the current risk posture and chip-plan notes.
- `report`: fixture horizons, scouting freshness, and whether player data covers the squad/watchlist or every player.

The `FPL_BRAIN_CONFIG` environment variable may point to another JSON file. A missing configuration fails with a message telling you to copy `config.example.json`.

## Finding the Team ID

Open the FPL website, select the Points tab for the team, and read the integer in the URL. Put that integer in `fpl_entry_id`. The public entry and history endpoints are supported; authenticated `my-team/` data and login are intentionally out of scope.

## Commands

```bash
python scripts/fetch_fpl.py                         # bootstrap + all fixtures
python scripts/fetch_fpl.py --summaries squad       # detail for squad/watchlist
python scripts/fetch_fpl.py --summaries all         # detail fan-out for all players
python scripts/fetch_fpl.py --live --gw 1            # event live data
python scripts/fetch_fpl.py --dry-run                # fetch/parse without writes

python scripts/sync_manager.py                      # configured current/next GW
python scripts/sync_manager.py --entry 1234567 --event 1
python scripts/sync_manager.py --all-events

python scripts/import_scouting.py scouting/example_report.json
python scripts/import_scouting.py scouting/example_report.json --dry-run
python scripts/import_scouting.py scouting/example_report.json --force

python scripts/build_report.py                      # Markdown + JSON
python scripts/build_report.py --gw 1 --format md --out data/exports/custom.md

python scripts/notes.py watchlist add --player-id 1 --status WATCH --reason "Monitor"
python scripts/notes.py watchlist list
python scripts/notes.py decision add --event 1 --action CAPTAIN --captain-id 1 --confidence medium
python scripts/notes.py strategy set --risk-posture balanced
python scripts/db_shell.py
```

Every CLI supports `--config`, `--verbose`, and `--quiet`. Exit code `0` includes benign pre-season no-data states; `1` is a recoverable network/partial failure; `2` is configuration or usage failure.

## Post-GW Market Report V1

The Post-GW Market Report is a separate, read-only local analytics artifact. It consumes the existing SQLite FACT history only; it does not collect from the network, run migrations, or mutate scouting, strategy, decisions, manager state, watchlists, or the database. It opens SQLite with `mode=ro` plus `PRAGMA query_only=ON`; run it after the separate collector has finished so the local source state is coherent.

```bash
python scripts/build_market_report.py --gw 1
python scripts/build_market_report.py --gw 1 --dry-run
python scripts/build_market_report.py --gw 1 --dry-run --stdout markdown
python scripts/build_market_report.py --gw 1 --dry-run --stdout json --allow-incomplete
```

Normal output is written atomically to `reports/gwNN/post_gw_market_report.json` and `.md`. It refuses an incomplete or unverifiable target GW with exit code `4`. `--allow-incomplete` is intentionally restricted to `--dry-run --stdout`, which produces a clearly labelled preview and writes nothing.

`--config` keeps its existing meaning: it selects the core FPL Brain configuration and database path. Use `--market-config path/to/override.json` only for a strict override of `config/post_gw_market_report.default.json`; the resolved Market Report configuration is hashed into the output.
Completion finality requires `events.finished=true`, `events.data_checked=true`, every target-GW fixture to have `finished=1`, and no started-but-unfinished fixture. `finished_provisional=true` is retained as source metadata and is not an independent blocker once those finality conditions hold.

The report contains only GW Stars, xG/xA/xGI leaders, attacking residual signals, fixture-level Defensive Contributions, Minutes Watch, target-GW-only squad context, a capped statistical Scout candidate handoff, and data gaps/caveats. It makes no transfer, captaincy, or chip recommendation. Completed player performance is selected only through the shared `fixtures.finished=1` boundary, so schedule/preseeded `minutes=0` rows and null/sentinel fixture rows cannot become appearances or non-appearances.

The main report's future fixture horizons anchor at the earliest pending event at or after the report context. An old-event fixture is not automatically inserted into every horizon: once its official state proves it is pending, a known rescheduled kickoff is used only to place it within the relevant official event-deadline window. Its original FPL event is retained and the placement is derived for ordering/DGW accounting. A pending fixture with no safe placement (including a null event, unknown kickoff, or contradictory schedule) is excluded from numeric FDR and shown as a data gap. Completed and in-progress fixtures are excluded, and kickoff timestamps never infer state.

DEFCON uses only proven fixture-level `player_gameweeks.defensive_contribution` values. The current storage has no separately persisted official defensive-contribution-points field, so those points remain null and the whole section becomes `DATA GAP` when no per-fixture counts are available; it never reconstructs counts or points from snapshots, BPS, CBI, recoveries, or tackles. Our Squad Context likewise becomes `DATA GAP` when the exact `(entry_id, target_gw)` `squad_picks` snapshot is absent; it never substitutes `manager_state` or a latest squad.

## FACT / CONTEXT separation

Official FPL facts are written by fetch and manager sync to `events`, `teams`, `players`, `player_snapshots`, `fixtures`, `player_gameweeks`, `manager_state`, `squad_picks`, and `manager_chips`. Scouting is a separate append-only stream in `scouting_notes`; it never writes a scouting value into a fact table. Human strategy lives in `watchlist`, `decisions`, and `strategy`.

Reports label official facts `[FACT]`, researched beliefs `[SCOUTING]`, transparent calculations `[DERIVED]`, and manager-entered values `[MANUAL]`. Scouting values include their confidence and observation timestamp, and expired/old notes are marked `STALE`.

## Scouting format

`scouting/schema.json` accepts the canonical form with an `observations` array, and the shorthand form with flat keys such as `start_probability`, `expected_minutes`, `likely_role`, `rotation_risk`, and `five_gameweek_role_security`. Numeric values are stored in `value_num`; ordinal/text values are stored in `value_text`. Unknown keys are accepted with a warning and category `other` so scouting format drift does not silently disappear.

Names resolve by explicit `player_id`, full name, web name, unique substring, then optional team and position hints. Zero or ambiguous matches are rejected to `scouting/rejected/` while resolvable players in the same file still import. Imports are append-only; the same file hash requires `--force` to import again. `--dry-run` writes neither notes nor rejected files.

## Scouting workflow (Scout Operations V1)

The provenance split in practice:

| Layer | Label | Owner |
| --- | --- | --- |
| Official FPL API data | `[FACT]` | collector (`fetch_fpl.py`, `sync_manager.py`) |
| Researched football context | `[SCOUTING]` | external browser-capable AI scout under `LUNA_SCOUT_PROTOCOL_V1.md` |
| Transparent calculations | `[DERIVED]` | FPL Brain metrics |
| Watchlist, decisions, weekly plan | `[MANUAL]` | the manager |

FPL Brain prepares and ingests the research; it never performs the web research itself. The safe weekly sequence:

```bash
# 0. (once) configure fpl_entry_id, then collect facts
python scripts/fetch_fpl.py
python scripts/sync_manager.py

# 1. optional manual weekly input: edit config/scout_plan.json
#    (captaincy_candidates, transfer_candidates, extra_players, custom_questions)

# 2. generate the research brief for the scout
python scripts/build_scout_brief.py --gw 1

# 3. give data/scouting/briefs/gwNN_main_<ts>.md + LUNA_SCOUT_PROTOCOL_V1.md
#    to a browser-capable AI scout; save its JSON to data/scouting/inbox/

# 4. validate without writing anything
python scripts/validate_scouting.py data/scouting/inbox/<file>.json
python scripts/import_scouting.py data/scouting/inbox/<file>.json --dry-run

# 5. import (append-only; identical file hashes are blocked without --force)
python scripts/import_scouting.py data/scouting/inbox/<file>.json

# 6. rebuild the report
python scripts/build_report.py --gw 1
```

Operational diagnostics:

```bash
python scripts/scout_status.py --gw 1   # coverage, staleness, contradictions, priorities
```

The scout is agent-neutral: the JSON `agent` field records whether the file came from Luna, GLM, or any other scout. Missing manual inputs never crash the brief generator; the brief reports the gaps (`CAPTAINCY CANDIDATES: MANUAL INPUT REQUIRED`, and similar) instead of inventing candidates. The importer stores canonical observations' human-readable field under `observation` (a legacy `note` field is still accepted first), so protocol-canonical files import with their note text intact.

Manager strategy: `FPL_STRATEGY_V1_FINAL.md` — the frozen canonical manager-level decision framework (transfers, hits, chips, captaincy, mini-league play) that sits above FPL Brain and Scout Operations.

## Inspecting the database


The database is ordinary SQLite:

```bash
python scripts/db_shell.py
```

Useful checks include:

```sql
SELECT COUNT(*) FROM teams;
SELECT COUNT(*) FROM players WHERE is_active = 1;
SELECT COUNT(*) FROM fixtures WHERE event IS NULL;
SELECT player_id, COUNT(*) FROM player_snapshots GROUP BY player_id LIMIT 10;
SELECT * FROM v_scouting_current;
```

Raw successful API bodies are kept under `data/raw/<fetch_run_id>/`. The raw response is written by `api.py` before parsing, so an unknown future field remains recoverable. `data/raw/fetch.log` records request endpoint, status, elapsed time, and stage counts without logging response bodies.

## API Notes

The required live inspection pass was run on 2026-08-19 against the 2026/27 public API using a normal `User-Agent` and no `Range` header. The committed files under `tests/fixtures/` are trimmed real responses for offline tests.

- `bootstrap-static/` returned HTTP 200 and about 1.56 MB, with top-level keys `chips`, `events`, `game_settings`, `game_config`, `phases`, `teams`, `total_players`, `element_stats`, `element_types`, and `elements`. The live response contained 20 teams and 592 active elements at inspection time.
- Live element objects contained `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`, all four requested `_per_90` variants, `defensive_contribution`, `clearances_blocks_interceptions`, `recoveries`, `tackles`, and `opta_code`. These names are mapped only in `parsers.py`.
- `element_types` contained IDs 1–4 (`GKP`, `DEF`, `MID`, `FWD`); there was no manager position ID 5. The schema keeps the position table general but does not invent a fifth position.
- `fixtures/` returned 380 fixtures. At inspection time no fixture had `event IS NULL` and every `stats[]` array was empty because no match had been played. The ingest still stores nullable events and raw stats, so later postponed fixtures and finished-match stats are not discarded.
- `element-summary/1/` returned 38 scheduled fixtures, an empty current `history[]`, and five `history_past[]` rows. Current performance field names could not be inspected because the season had not started; the parser remains defensive and stores raw rows.
- `event/1/live/` returned exactly `{"elements": []}`. This is valid pre-season input. Per-GW reliability ignores scheduled summary rows that have no performance minutes, so the report still says that gameweek data is missing.
- `entry/1/` was public and included `last_deadline_bank`, `last_deadline_value`, and `last_deadline_total_transfers`. The first two were null pre-season, while the last was `0`; summary points/rank/event fields were null. The implementation stores null bank/value rather than inventing values.
- The public entry response's `name` is the FPL team name. Personal manager identity is carried separately by `player_first_name` and `player_last_name`; reports label both fields separately.
- `entry/1/history/` returned an empty `current[]`, populated `past[]`, and empty `chips[]`. The historical rows exposed `rank` rather than `overall_rank`; the parser accepts both forms. No current transfer fields could be observed while `current[]` was empty.
- `entry/1/event/1/picks/` returned HTTP 404 before the deadline. `FplNotFoundError` is converted to “no data yet”; manager sync exits successfully and does not create squad picks.
- `event/<gw>/live/` represents double-gameweek fixture detail through each player's `explain[]` entries. The parser stores one row per distinct explained fixture and does not copy event-total `stats` into every fixture row; missing fixture-level values remain null and the raw live body is retained.
- The live chip definitions used `start_event=2` for the first wildcard and free hit, while bench boost and triple captain began at GW1. The report shows each chip’s API event range rather than inventing a different start range.
- The live `bootstrap-static` payload also included `game_settings`, `game_config`, `phases`, `element_stats`, and `total_players`. They are retained in raw JSON but only the V1 tables/fields in the specification are persisted.
- `team/set-piece-notes/` was not integrated. It remains a V1.1 candidate as directed by the specification.

## Deviations

- Scouting validation is implemented with a small in-project validator rather than adding `jsonschema`; the accepted canonical/shorthand shape and permissive unknown-key behavior are the same for V1.
- `element-summary.fixtures[]` rows are stored as source=`element_summary` schedule rows so their fixture IDs, opponents, and kickoff times remain available. They are not counted as appearances or minutes in `minutes_reliability` until performance minutes exist.
- Gameweek upserts use explicit source precedence: schedule-only rows enrich metadata without clearing performance, completed summary rows may replace/enrich live rows, and a real completed fixture row removes only its event-live sentinel. Reliability selects the most recent completed events, never future schedule rows.
- Squad picks are an exact `(entry_id, event)` roster replacement when a non-empty picks response is available; empty or 404 responses retain the previous roster.
- A malformed or empty bootstrap is rejected before absence marking, so a failed parse cannot deactivate every player. The latest fetch status is shown in reports and partial/failed runs are flagged as potentially stale FACT data.
- The report uses the live API’s chip ranges and names directly. This means the pre-season report can show `GW2–19` for the first wildcard/free hit when that is what the live response says.

## Verification

The offline test suite uses no network and passes with `python -m pytest -q`. The end-to-end live checks completed on 2026-08-19 included two successful fetches (592 snapshots each generation), raw-response persistence, a 380-fixture database, null Team ID handling, public Team ID 1 manager sync, graceful GW1 picks 404 handling, dry-run fetch, scouting dry-run/import, and Markdown/JSON report generation.
