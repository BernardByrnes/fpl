# FPL Brain — Luna Scout Integration Handoff

## Status and scope

This document describes the implemented FPL Brain scouting integration as inspected on 2026-08-19. It is the handoff contract for an external AI architect designing the Luna web-scouting methodology and operating protocol. It is documentation only: Luna must produce imports compatible with the existing implementation and must not redesign FPL Brain.

## 1. Purpose of FPL Brain

FPL Brain is a local Python + SQLite intelligence pipeline for the 2026/27 Fantasy Premier League season. It collects official FPL facts, keeps historical snapshots and gameweek data, imports external scouting beliefs, and generates Markdown and JSON briefings for downstream LLM decision-making and manager review.

The system intentionally keeps official facts, external scouting, transparent calculations, and manager-entered decisions distinguishable. It does not produce a single opaque player score.

## 2. Provenance model

The generated report uses four provenance labels:

- [FACT] — data obtained from the official FPL API and stored by the collector, including current values and historical snapshots.
- [SCOUTING] — an externally researched or inferred football observation imported through the scouting layer. It is a belief with confidence and timing, not an official FPL fact.
- [DERIVED] — a calculation made by FPL Brain from stored inputs, such as movement, reliability, fixture summaries, or other transparent comparisons.
- [MANUAL] — manager-entered state or strategy, such as watchlist entries, decisions, or configuration.

Luna scouting may write only to the scouting/context layer: the scouting_imports and scouting_notes tables, through scripts/import_scouting.py. Luna must never manufacture official player, fixture, price, ownership, points, status, news, team-strength, or other FPL facts. If a fact is not supplied by the official FPL API collector, it must not be presented as [FACT] merely because a web source mentions it.

## 3. Official information FPL Brain already provides

Luna does not need to research or restate the following structured FPL information when the relevant official records are present:

- FPL player IDs, names, web names, teams, positions, prices, ownership, status, and official news/news dates;
- official transfer totals and current points;
- minutes, goals, assists, and other stored player-gameweek performance fields;
- xG, xA, and xGI when supplied by the official API payload;
- fixtures, gameweeks, difficulty ratings/FDR, kickoff/status fields, and fixture completion state;
- official team strength fields collected for teams;
- historical player and team snapshots, including the fields stored by the snapshot tables;
- configured current manager state, squad picks, chips, and active manual watchlist data.

The exact official field coverage follows the collector and schema. Some fields are nullable or can be absent in a partial/failed fetch; absence is a data gap, not permission for Luna to fill the value as an official fact. Luna’s job is to add current football context that the FPL API does not provide reliably.

## 4. What Luna scouting is for

Luna supplies uncertain, current, football-context observations that are useful alongside official FPL facts, for example:

- expected start probability and expected minutes;
- likely tactical role;
- penalty, corner, and free-kick responsibility;
- rotation risk and five-gameweek role security;
- competition for the position;
- fitness uncertainty;
- transfer-exit risk;
- European/fixture congestion;
- tactical observations;
- contextual attacking-threat and DEFCON potential.

These are observations, not guarantees. Luna should include the observation time, confidence, and evidence needed for a later reviewer to understand why the belief was recorded.

## 5. Exact scouting JSON contract

### Root object

The runtime validator requires:

- schema_version: exactly the string "1.0";
- players: an array.

The implemented schema also permits generated_at, agent, gameweek, and default_expires_at. The JSON Schema describes their intended types: generated_at and default_expires_at are strings (default_expires_at may be null), agent is a string, and gameweek is an integer at least 1. The runtime validator does not deeply validate every optional root type, so Luna should still emit the documented types.

### Player object

Every player entry requires a non-empty player_name. It may contain:

- player_id: integer or null;
- team_hint: string;
- position_hint: string;
- confidence: low, medium, or high;
- summary: a player-level note inherited by shorthand observations;
- evidence: a player-level evidence array inherited by shorthand observations;
- observations: canonical observations, or shorthand fields.

### Canonical observation object

An observation object requires:

- key: non-empty string;
- value: non-null value.

It may also contain category, unit, confidence, evidence, observation, gameweek_context, observed_at, and expires_at. Observation confidence, when present, must be low, medium, or high.

An observations list may contain observation objects or strings. A string is imported as a tactical_note. If any observation in the list is an object, the list is treated as canonical; canonical observations cannot be mixed with arbitrary player-level shorthand fields. A player using shorthand must not provide null values for supported shorthand keys.

### Canonical form: complete valid example

    {
      "schema_version": "1.0",
      "generated_at": "2026-08-19T09:00:00Z",
      "agent": "luna-scout",
      "gameweek": 1,
      "default_expires_at": "2026-09-02T23:59:59Z",
      "players": [
        {
          "player_id": 123,
          "player_name": "Example Player",
          "team_hint": "Example FC",
          "position_hint": "MID",
          "confidence": "medium",
          "summary": "Role appears stable but minutes are not guaranteed.",
          "observations": [
            {
              "key": "start_probability",
              "value": 75,
              "category": "minutes",
              "unit": "percent",
              "confidence": "medium",
              "evidence": [
                {
                  "source": "Example FC press conference",
                  "source_type": "club",
                  "url": "https://example.com/source",
                  "date": "2026-08-18",
                  "quote": "Example quote or concise supporting note"
                }
              ],
              "observation": "Expected to start if fit.",
              "gameweek_context": 1,
              "observed_at": "2026-08-19T09:00:00Z",
              "expires_at": "2026-09-02T23:59:59Z"
            },
            {
              "key": "likely_role",
              "value": "advanced central midfielder",
              "category": "role",
              "confidence": "medium",
              "observation": "Often arrives in the box from the left half-space."
            }
          ]
        }
      ]
    }

The example uses an explicit ID and shows the optional evidence/timing fields. source_type, url, date, and quote are useful evidence metadata but are not separately enforced by the runtime validator; see Evidence format below.

### Shorthand form

In shorthand, fields directly under a player other than player metadata are converted into observations. Recognised aliases are canonicalised. Player confidence, summary, and evidence provide defaults. Root generated_at, gameweek, default_expires_at, and agent provide defaults where observation-level values are absent.

### Shorthand form: complete valid example

    {
      "schema_version": "1.0",
      "generated_at": "2026-08-19T09:00:00Z",
      "agent": "luna-scout",
      "gameweek": 1,
      "default_expires_at": "2026-09-02T23:59:59Z",
      "players": [
        {
          "player_name": "Example Player",
          "team_hint": "Example FC",
          "position_hint": "MID",
          "confidence": "low",
          "summary": "Monitor rotation after the next European match.",
          "evidence": [
            {
              "source": "Trusted beat reporter",
              "source_type": "journalist",
              "url": "https://example.com/report",
              "date": "2026-08-19",
              "notes": "Concise source note"
            }
          ],
          "start_probability": 60,
          "expected_minutes": 68,
          "rotation_risk": "medium",
          "role_security_5gw": "high",
          "likely_role": "wide forward",
          "set_piece_role": "corners",
          "tactical_note": "May be protected when the team leads."
        }
      ]
    }

In shorthand, tactical_note is a recognised key. A string in an observations list is also treated as a tactical note. Do not combine canonical observation objects with shorthand player fields.

## 6. Controlled scouting vocabulary

The runtime recognises the following keys. The category and unit shown are the defaults applied when the observation does not explicitly provide them.

| Key | Category | Expected type | Default unit | Accepted range/ordinal values |
|---|---|---|---|---|
| start_probability | minutes | numeric | percent | 0–100 |
| expected_minutes | minutes | numeric | minutes | 0–90 |
| rotation_risk | minutes | ordinal/text | scale | ordinal values are stored as text; no runtime enum enforcement |
| role_security_5gw | minutes | ordinal/text | scale | ordinal values are stored as text; no runtime enum enforcement |
| competition_for_position | minutes | ordinal/text | scale | ordinal values are stored as text; no runtime enum enforcement |
| european_congestion | minutes | ordinal/text | scale | ordinal values are stored as text; no runtime enum enforcement |
| penalty_probability | setpieces | numeric | percent | 0–100 |
| freekick_probability | setpieces | numeric | percent | 0–100 |
| corner_probability | setpieces | numeric | percent | 0–100 |
| set_piece_role | setpieces | text | none | free text |
| injury_uncertainty | risk | numeric | percent | 0–100 |
| transfer_exit_risk | risk | numeric | percent | 0–100 |
| likely_role | role | text | none | free text |
| attacking_threat | role | ordinal/text | scale | ordinal values are stored as text; no runtime enum enforcement |
| defcon_potential | role | ordinal/text | scale | ordinal values are stored as text; no runtime enum enforcement |
| tactical_note | other | text | none | free text |

The implemented alias five_gameweek_role_security is canonicalised to role_security_5gw. Numeric keys are converted to value_num where possible. For numeric keys, the importer warns when the value is outside the ranges above but still imports it; it does not reject the note. Non-numeric values for numeric keys are warned about and stored as text. Other numeric values use value_num; other values use value_text.

The importer also accepts an explicit category and unit, so the defaults above can be overridden. CATEGORY_DEFAULTS maps the keys exactly as listed; there are no special runtime categories named transfer or fitness.

Unknown keys are currently accepted with a warning, assigned category other unless a category is supplied, and imported. This is compatibility behaviour, not a signal that an unknown key is part of the controlled vocabulary. An unknown key should not be used to manufacture an official fact.

## 7. Evidence format

The importer accepts evidence on an observation or at player level. It also accepts root/player/observation timing and provenance fields:

- evidence: normally an array of JSON objects or values; it is serialized and stored as evidence_json without extracting or validating a fixed sub-schema;
- source, source_type, url, date, quote, and notes: useful metadata inside each evidence object, preserved as supplied;
- confidence: low, medium, or high at player or observation level;
- observed_at: ISO-like timestamp for when the belief was observed;
- expires_at: ISO-like timestamp after which the observation is stale;
- generated_at: root fallback for observed_at;
- default_expires_at: root fallback for expires_at;
- observation: concise human-readable explanation;
- agent: root provenance string stored on imported notes.

The exact evidence payload is pass-through JSON. URLs should be complete web URLs; source should identify the publication/person/club; source_type should identify the source kind; date should be the source publication or event date; quote or notes should preserve a short supporting detail. These conventions improve auditability, but the current runtime does not require those keys or verify that a URL resolves.

## 8. Confidence, freshness, and staleness

Confidence is represented as lowercase low, medium, or high. The importer applies observation confidence first, then player confidence, then low as the fallback. Invalid supplied confidence values raise a validation/import error.

observed_at is stored on every note and is required by the database. The importer uses observation-level observed_at, then root generated_at, then the current UTC time. expires_at uses observation-level expires_at, then root default_expires_at, otherwise it remains null. Timestamps are stored as supplied after the importer’s normalisation path; report freshness parsing expects ISO-style values.

Reports use the configured scouting_stale_after_days, which defaults to 14 days. A current note is marked STALE when either:

1. its expires_at is in the past; or
2. its observed_at is older than the configured stale-after interval.

Stale notes remain visible in SCOUTING CONTEXT; staleness is a warning, not automatic deletion. The current note for each (player_id, key) is selected by observed_at descending, with the database note id descending as the deterministic tie-break. This means a newer belief supersedes an older belief for current display, even though the older row remains in history.

## 9. Append-only scouting history

Every successful import creates an audit row in scouting_imports; every normalised observation creates a row in scouting_notes. Notes are not updated in place and newer beliefs do not destroy previous observations. The current compatibility/ranked view uses a row number partitioned by player and key, ordered by observed_at descending, then id descending, and exposes rank 1 to the repository/report.

This permits later review of what Luna believed, when it believed it, which import supplied it, and what evidence was attached. A correction should be a new observation, not an edit to an old one.

## 10. Player resolution

The implemented resolver operates against active FPL players and follows this order:

1. An explicit player_id, when supplied and resolvable, wins immediately.
2. Otherwise names are normalised using Unicode accent stripping, case folding, punctuation removal, and whitespace collapsing.
3. Exact normalised full-name matches are considered before exact web-name matches.
4. If needed, normalised substring matches across full name/web name are considered.
5. If multiple candidates remain, team_hint is matched against normalised team name or short name.
6. If ambiguity remains, position_hint is matched against the normalised position short name.

Exactly one candidate must remain. No candidate or more than one candidate is unresolved. Luna must never guess an ambiguous identity. Include the explicit FPL player ID whenever available; otherwise provide enough team and position hints to make the identity unambiguous. Resolution uses active candidates for normal name-based scouting imports.

## 11. Import behaviour

The CLI is:

    python scripts/import_scouting.py PATH_TO_SCOUTING_JSON

Useful options are:

    python scripts/import_scouting.py PATH_TO_SCOUTING_JSON --dry-run
    python scripts/import_scouting.py PATH_TO_SCOUTING_JSON --force
    python scripts/import_scouting.py PATH_TO_SCOUTING_JSON --verbose

- Normal import validates the document, resolves players, writes an append-only import audit row and resolved notes, and prints scouting success with resolved/total players, note count, and unresolved count.
- --dry-run validates and resolves without writing notes, imports, or rejected-player files. It reports status dry-run and is useful before handing a file to the live database.
- The file SHA-256 is recorded. Re-importing the same file is blocked as a duplicate unless --force is supplied. --force permits another append-only import of the same content; it does not update old rows.
- Unresolved players are written to a rejected JSON file under the configured scouting/rejected location, with source-parent handling implemented by the importer. The rejected entry includes unresolved player context for correction.
- Resolvable players are imported even when some players are unresolved. The import audit records total, resolved, inserted-note counts, and unresolved JSON. Thus an import can be partial; Luna should inspect the CLI result and rejected file.
- Missing configuration/database, schema/JSON validation errors, duplicate imports, and other import failures are surfaced by the CLI with non-zero exit status. A duplicate is exit 1; configuration errors are exit 2.

The default project examples are scouting/example_report.json and the commands documented in README.md.

## 12. Report consumption

Report selection is controlled by configuration. By default it is squad_and_watchlist: selected players are the configured current squad plus active manual watchlist players, without duplicates. If all-player selection is configured, the report can include all active players. Scouting is not a substitute for configuring the selected population.

### SCOUTING CONTEXT

For each selected player with a current scouting note, the Markdown report emits the player/team and lines in this shape:

    SCOUTING CONTEXT
      Example Player (Example FC)
        [SCOUTING] start_probability: 75 (confidence medium, observed 2026-08-19T09:00:00Z)
          Note: Expected to start if fit.

The report chooses value_text when present, otherwise value_num, and includes confidence and observed timestamp. It adds STALE when the freshness rules above apply. It does not currently print evidence_json, category, unit, or expires_at as separate report lines, although those values remain stored in SQLite and are available to tooling/report JSON as implemented fields where applicable. If no selected player has notes, it says: No scouting notes for selected players.

The JSON report mirrors the section and line structure rather than exposing a different scouting contract. Downstream LLMs should treat these lines as uncertain context and not as official facts.

### RISKS

RISKS combines official status/news facts with selected scouting risk notes. Scouting notes are included when their key is one of:

    rotation_risk
    transfer_exit_risk
    role_security_5gw
    injury_uncertainty

or when their confidence is low. The rendered line follows the same [SCOUTING] player key: value (confidence ..., observed ...) pattern. This is a risk-focused projection of the current notes, not a second source of facts.

### WATCHLIST

WATCHLIST is labelled [MANUAL + FACT] and is driven by the configured active manual watchlist. It reports manager-entered reason/target gameweek and official current price, ownership, and fixture/FDR information where available. Scouting notes are not directly rewritten as watchlist decisions. A watchlist player is nevertheless included in the default squad-and-watchlist scouting selection.

### DATA GAPS AND CAVEATS

The report flags missing/partial official fetches and missing gameweek performance data, and it flags missing scouting notes for selected players. If no manager entry is configured, manager/squad sections are limited accordingly. A pre-season or partial season is not evidence that a missing official field should be researched and inserted by Luna.

## 13. Current scouting-integration limitations

- The scout/importer does not write to the official FPL tables or to FPL itself.
- FPL Brain contains no automatic web-research agent. Luna’s research and the import file are an external/manual import workflow.
- There is no ML player score or opaque composite ranking in the implementation.
- The official set-piece endpoint is not integrated; set-piece observations must come from Luna’s scouting evidence and remain [SCOUTING].
- Evidence metadata is stored as pass-through JSON; the runtime does not verify URLs, dates, quotes, or source credibility.
- Unknown observation keys are accepted with warnings, so schema discipline must be enforced by Luna’s operating protocol.
- Stale observations remain in current-context reports with a STALE marker; they are not automatically removed.
- Report display is intentionally compact: evidence, unit, category, and expiry are stored but are not all rendered as dedicated lines.

## 14. Scout design constraints

The external architect should design the research methodology and Luna operating protocol only. It must not redesign:

- FPL Brain;
- the SQLite schema;
- the official API collector;
- the report architecture;
- the dashboard.

Any proposed Luna output must be valid for the implemented scouting importer: root schema_version "1.0", a players array, resolvable identities, recognised keys where possible, valid confidence values, and explicit timing/evidence suitable for append-only observations. Luna must preserve the provenance boundary and never emit scouting beliefs as official FPL facts.

## 15. Relevant implementation references

The following files were inspected to produce this handoff:

- K:\FPL\fpl_brain\scouting.py — runtime validation, shorthand expansion, key defaults, evidence/timestamp normalisation, resolution, and import flow;
- K:\FPL\scouting\schema.json — declared JSON Schema for the scouting document;
- K:\FPL\fpl_brain\database.py — scouting tables and current/ranked views;
- K:\FPL\fpl_brain\repositories.py — player resolution candidates, scouting inserts, and current-note query;
- K:\FPL\fpl_brain\reports.py — report sections, current-note rendering, stale logic, risk projection, and data-gap lines;
- K:\FPL\fpl_brain\config.py — scouting freshness and player-selection defaults;
- K:\FPL\fpl_brain\utils.py — name and timestamp normalisation used by the importer;
- K:\FPL\scripts\import_scouting.py — command-line interface and exit behaviour;
- K:\FPL\scouting\example_report.json — project example of canonical and shorthand input;
- K:\FPL\README.md — project workflow, provenance description, commands, and API notes.

## README/specification discrepancies recorded for the architect

The implemented behaviour, not the older specification or illustrative README examples, is authoritative:

1. The old API-note wording says minutes reliability ignores scheduled summary rows that have no performance minutes. The implemented logic now joins player_gameweeks.fixture_id to fixtures.id and includes only rows whose fixture has fixtures.finished = 1; a future row with minutes = 0 is therefore excluded, while a genuinely completed fixture with zero minutes is eligible.
2. The old controlled-vocabulary description suggests ordinal values such as very_low through very_high. The runtime stores ordinal keys as text and does not enforce that enum, so the architect must choose and consistently emit the project’s intended vocabulary rather than assume runtime rejection of other strings.
3. Older examples describe evidence fields such as source, url, date, and quote as if they were a strict schema. The implementation preserves evidence JSON but does not validate a required evidence sub-schema or verify URLs.
4. Older report examples show a dedicated Evidence line in SCOUTING CONTEXT. The implemented report currently renders value, confidence, observed time, stale status, and optional note; it does not render evidence fields as dedicated lines.
5. Older category examples include transfer and fitness. The implemented default categories are minutes, setpieces, risk, role, and other; transfer_exit_risk and injury_uncertainty default to risk.
6. Current-note selection is deterministic observed_at descending, then id descending, not merely an unspecified maximum timestamp. Historical note rows remain append-only.

