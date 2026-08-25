"""Scout Operations V1: research-brief generation over existing database reads.

This module prepares work for an external browser-capable AI scout operating
under LUNA_SCOUT_PROTOCOL_V1.md.  It never performs research itself and never
recommends a transfer, captain, chip, or ranking; it states what needs
investigation, not what the answer is.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import repositories as repo
from .utils import parse_utc, utc_now

PROTOCOL_FILENAME = "LUNA_SCOUT_PROTOCOL_V1.md"

# Tier caps from protocol section A1 (players per tier) plus the global hard
# cap of 40 players for a main pass.  Per-tier maxima and the overall cap are
# independent: 15+5+8+8+8=44 candidates can be legal per tier yet must still
# be trimmed to 40 overall, lowest-priority tier first.
TIER_CAPS = {1: 15, 2: 5, 3: 8, 4: 8, 5: 8}
PLAYER_CAP = 40

FLAG_LABELS = {
    "squad": "CURRENT SQUAD",
    "captaincy": "CAPTAINCY CANDIDATE",
    "transfer_manual": "TRANSFER TARGET",
    "watchlist_buy": "WATCHLIST BUY",
    "trigger": "TRIGGER SITUATION",
}

# Explicit-signal keywords.  A missing risk key is only surfaced as an open
# question when the manual plan or a watchlist reason explicitly asks about
# that domain; absence alone is never treated as a problem (protocol A4/A17).
INJURY_KEYWORDS = ("injur", "fitness", "knock", "illness")
SET_PIECE_KEYWORDS = ("penalt", "set piece", "set-piece", "set_piece", "free kick", "free-kick", "freekick", "corner")
SET_PIECE_NOTE_KEYS = ("penalty_probability", "freekick_probability", "corner_probability", "set_piece_role")
TRANSFER_KEYWORDS = ("transfer", "exit risk", "departure", "move away")
COMPETITION_KEYWORDS = ("competition", "competitor", "rival for", "battle for")

# Keys the open-question logic treats as the core minutes/role/set-piece view.
CORE_NOTE_KEYS = (
    "start_probability",
    "expected_minutes",
    "likely_role",
    "role_security_5gw",
)
SET_PIECE_KEYS = ("penalty_probability", "set_piece_role")
RISK_KEYS = (
    "rotation_risk",
    "role_security_5gw",
    "injury_uncertainty",
    "transfer_exit_risk",
)
CONTRADICTION_TOKEN = "CONTRADICTION:"
UNRESOLVED_TOKEN = "UNRESOLVED:"

CONDENSED_INSTRUCTIONS = [
    "Full methodology: LUNA_SCOUT_PROTOCOL_V1.md (sections A1-A17). Condensed execution rules:",
    "1. You are a scout, not the manager: never recommend buy/sell/captain/chip. Report evidence only.",
    "2. Never restate official FPL facts (price, ownership, points, xG, fixtures, FDR, official news) as observations.",
    "3. Sources in tier order: official club/manager > competitive match records > embedded beat/injury",
    "   specialists > reputable national/tactical/predicted lineups > informed community; social media never",
    "   sole source. Facts go in evidence; inferences go in values.",
    "4. Probabilities use ONLY bands 95/85/70/50/30/15/5; ordinals very_low/low/medium/high/very_high;",
    "   confidence strictly low/medium/high (weakest evidence link sets the ceiling; evidence-driven, no quota).",
    "5. expected_minutes is transparent arithmetic from start probability, minutes-if-starting anchor,",
    "   early-hook risk, and bench contribution; state the arithmetic in the observation.",
    "6. Preserve contradictions: one note per key, value toward neutral, low confidence, both sources in",
    "   evidence, observation prefixed CONTRADICTION:. UNKNOWN is complete: omit the key, mark UNRESOLVED:.",
    "7. Stop per protocol A15, including the decisive-primary-source immediate stop for official",
    "   injury/suspension/unavailability/registration facts. Search budgets: 6/4/2 by tier.",
    "8. Output: valid FPL Brain scouting JSON (canonical form, 16 controlled keys, one note per",
    "   (player,key), self-contained observation text, evidence array, observed_at/expires_at) plus a",
    "   plain-text research log. Then run the Pre-Output QC Checklist (protocol section F).",
]


def load_scout_plan(path: str | Path | None) -> tuple[dict[str, Any], list[str]]:
    """Load the optional manual weekly plan; a missing file is an empty plan."""

    if not path:
        return {}, []
    source = Path(path)
    if not source.exists():
        return {}, [f"scout plan {source} not found — using empty plan"]
    try:
        plan = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"scout plan {source} unreadable ({exc}) — using empty plan"]
    if not isinstance(plan, dict):
        return {}, [f"scout plan {source} must be a JSON object — using empty plan"]
    return plan, []


def _player_label(row: dict[str, Any]) -> str:
    return str(row.get("web_name") or row.get("full_name") or f"Player {row.get('player_id') or row.get('id')}")


def _note_text(note: dict[str, Any]) -> str:
    value = note.get("value_text") if note.get("value_text") is not None else note.get("value_num")
    return "—" if value is None else str(value)


def _note_status(
    note: dict[str, Any],
    now_dt: Any,
    deadline_dt: Any,
    stale_days: int,
) -> str:
    observed_dt = parse_utc(note.get("observed_at"))
    expires_dt = parse_utc(note.get("expires_at"))
    if (expires_dt is not None and expires_dt < now_dt) or (
        observed_dt is not None and (now_dt - observed_dt).days > stale_days
    ):
        return "STALE"
    if expires_dt is not None and deadline_dt is not None and expires_dt <= deadline_dt:
        return "expires before deadline"
    return "current"


def _watchlist_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """SELECT w.*, p.web_name, p.full_name, t.name AS team_name, t.short_name AS team_short_name,
                      pos.singular_name_short AS position_short_name
                 FROM watchlist w JOIN players p ON p.id=w.player_id
                 LEFT JOIN teams t ON t.id=p.team_id
                 LEFT JOIN positions pos ON pos.id=p.element_type
                WHERE w.is_active=1 ORDER BY w.id"""
        ).fetchall()
    ]


def _player_row(conn: sqlite3.Connection, player_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT p.id AS player_id, p.web_name, p.full_name, t.name AS team_name,
                  t.short_name AS team_short_name, pos.singular_name_short AS position_short_name
             FROM players p LEFT JOIN teams t ON t.id=p.team_id
             LEFT JOIN positions pos ON pos.id=p.element_type
            WHERE p.id=? AND p.is_active=1""",
        (player_id,),
    ).fetchone()
    return dict(row) if row else None


def _plan_player_ids(plan: dict[str, Any], field: str) -> list[int]:
    ids: list[int] = []
    for value in plan.get(field) or []:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def _resolve_plan_players(
    conn: sqlite3.Connection,
    player_ids: list[int],
    warnings: list[str],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for player_id in player_ids:
        row = _player_row(conn, player_id)
        if row is None:
            warnings.append(f"{label}: player_id {player_id} not found among active players — ignored")
        elif row not in rows:
            rows.append(row)
    return rows


def _has_signal(signals: list[str], keywords: tuple[str, ...]) -> bool:
    return any(keyword in signal.lower() for signal in signals for keyword in keywords)


def _has_set_piece_signal(
    signals: list[str],
    by_key: dict[str, dict[str, Any]],
    statuses: dict[str, str],
) -> bool:
    """True only when something actually asks for set-piece research.

    Valid signals: an explicit manual/custom/extra question or watchlist
    reason mentioning the set-piece domain, or an existing set-piece note
    that is stale, expiring before the deadline, low confidence, or carries
    a CONTRADICTION:/UNRESOLVED: marker (those surface their own refresh or
    reconciliation question via the per-note checks).  Absence of set-piece
    notes alone is never a signal — "unknown because nobody asked" is not a
    scouting gap.  No football judgement is inferred from position, team,
    or player name.
    """

    if _has_signal(signals, SET_PIECE_KEYWORDS):
        return True
    for key in SET_PIECE_NOTE_KEYS:
        note = by_key.get(key)
        if note is None:
            continue
        if note.get("confidence") == "low" or statuses.get(key) in {"STALE", "expires before deadline"}:
            return True
        text = f"{note.get('value_text') or ''} {note.get('observation') or ''}"
        if CONTRADICTION_TOKEN in text or UNRESOLVED_TOKEN in text:
            return True
    return False


def open_questions_for_player(
    notes: list[dict[str, Any]],
    now_dt: Any,
    deadline_dt: Any,
    stale_days: int,
    custom_questions: list[str] | None = None,
    signals: list[str] | None = None,
) -> list[str]:
    """Derive what needs investigation from missing, stale, expiring, low-confidence,
    contradictory, or unresolved current notes.  States questions; never answers them.

    Missing risk keys (injury, transfer, competition) are surfaced only when an
    explicit signal — a manual question or watchlist reason — asks about that
    domain.  Absence of a note alone is not evidence of a problem.
    """

    questions: list[str] = []
    custom_questions = [q for q in (custom_questions or []) if isinstance(q, str) and q.strip()]
    signals = list(signals or []) + custom_questions
    by_key: dict[str, dict[str, Any]] = {str(note["key"]): note for note in notes}
    statuses = {key: _note_status(note, now_dt, deadline_dt, stale_days) for key, note in by_key.items()}
    set_piece_present = set(SET_PIECE_NOTE_KEYS) & set(by_key)

    if not by_key:
        questions.append("no scouting notes at all — establish minutes, role, and set-piece view")
    else:
        for key in CORE_NOTE_KEYS:
            if key not in by_key:
                questions.append(f"{key} missing — no current scouting view for this player")
        if not set_piece_present and _has_set_piece_signal(signals, by_key, statuses):
            questions.append(
                "set-piece hierarchy unknown — set-piece research is requested but no penalty/free-kick/corner/set_piece_role note exists"
            )
        if "rotation_risk" not in by_key and ({"start_probability", "expected_minutes"} & set(by_key)):
            questions.append("rotation_risk absent although a minutes view exists — add the 2-3 GW view")

    if set_piece_present and _has_signal(signals, SET_PIECE_KEYWORDS):
        missing = [key for key in SET_PIECE_NOTE_KEYS if key not in by_key]
        if missing:
            questions.append("set-piece coverage incomplete despite an explicit set-piece question — missing " + ", ".join(missing))
    if "injury_uncertainty" not in by_key and _has_signal(signals, INJURY_KEYWORDS):
        questions.append("injury_uncertainty missing despite an explicit injury/fitness question — establish the fitness view")
    if "transfer_exit_risk" not in by_key and _has_signal(signals, TRANSFER_KEYWORDS):
        questions.append("transfer_exit_risk missing despite an explicit transfer question — establish the exit-risk view")
    if "competition_for_position" not in by_key and _has_signal(signals, COMPETITION_KEYWORDS):
        questions.append("competition_for_position missing despite an explicit competition question — establish the competition view")

    for key, note in sorted(by_key.items()):
        status = statuses[key]
        if status == "STALE":
            questions.append(f"{key} is STALE (observed {note.get('observed_at')}) — re-affirm or refresh")
        elif status == "expires before deadline":
            questions.append(f"{key} expires {note.get('expires_at')} — before the GW deadline; refresh needed")
        if note.get("confidence") == "low":
            questions.append(f"{key} has low confidence — verify with stronger evidence or record why not")

    # Literal protocol markers are scanned across every current note's stored
    # value and observation text, not only tactical_note.  Contradictions are
    # never inferred from differing numbers.
    all_note_text = " ".join(f"{note.get('value_text') or ''} {note.get('observation') or ''}" for note in notes)
    if CONTRADICTION_TOKEN in all_note_text:
        questions.append("a current note records a CONTRADICTION — reconcile the conflicting evidence per protocol A14")
    if UNRESOLVED_TOKEN in all_note_text:
        questions.append("a current note records an UNRESOLVED item — this is the top of the research queue")

    questions.extend(custom_questions)

    if not questions:
        questions.append("no open question — cheap re-affirmation only (protocol A1)")
    return questions


def _is_research_needed(questions: list[str]) -> bool:
    return not (len(questions) == 1 and questions[0].startswith("no open question"))


def _apply_global_player_cap(tiers: dict[str, list[dict[str, Any]]], warnings: list[str]) -> None:
    """Enforce the protocol A1 global hard cap of PLAYER_CAP players.

    Applied after the independent per-tier caps.  Trimming is deterministic:
    players are removed from the end of the lowest-priority tier upward
    (Tier 5, then 4, 3, 2, and only then 1), preserving each tier's existing
    order.  Markdown brief, JSON brief, questions, counts, and status all
    read the same post-cap tiers.
    """

    total = sum(len(entries) for entries in tiers.values())
    if total <= PLAYER_CAP:
        return
    excess = total - PLAYER_CAP
    omitted: list[str] = []
    for tier in (5, 4, 3, 2, 1):
        entries = tiers[str(tier)]
        while excess and entries:
            dropped = entries.pop()
            omitted.append(f"Tier {tier} {dropped['name']} (ID {dropped['player_id']})")
            excess -= 1
        if not excess:
            break
    warnings.append(
        f"GLOBAL CAP APPLIED: {len(omitted)} players omitted to keep the research universe at {PLAYER_CAP}: "
        + "; ".join(omitted)
    )


def build_brief(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    gw: int | None = None,
    plan: dict[str, Any] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Assemble the structured research brief from the database and manual plan."""

    plan = plan or {}
    now_str = now or utc_now()
    now_dt = parse_utc(now_str)
    event_id = int(gw) if gw else repo.current_or_next_event(conn)
    events = {int(row["id"]): row for row in repo.event_rows(conn)}
    event = events.get(event_id)
    deadline = event.get("deadline_time") if event else None
    deadline_dt = parse_utc(deadline)
    stale_days = int(config.get("report", {}).get("scouting_stale_after_days", 14))
    warnings: list[str] = []
    manual_gaps: list[str] = []

    entry_id = config.get("fpl_entry_id")
    squad = repo.squad_rows(conn, int(entry_id), event_id) if entry_id is not None else []
    if entry_id is None:
        manual_gaps.append("SQUAD: fpl_entry_id is not configured — set it in config.json and run sync_manager.py")
    elif not squad:
        manual_gaps.append(
            f"SQUAD: no squad picks stored for GW{event_id} — the deadline has not passed or sync_manager.py has not run"
        )
    watchlist = _watchlist_rows(conn)

    tiers: dict[str, list[dict[str, Any]]] = {str(tier): [] for tier in sorted(TIER_CAPS)}
    player_flags: dict[int, list[str]] = {}
    extra_questions: dict[int, list[str]] = {}

    def add_to_tier(tier: int, row: dict[str, Any], flag: str) -> None:
        player_id = int(row["player_id"] if "player_id" in row else row["id"])
        label = FLAG_LABELS.get(flag, flag)
        if player_id in player_flags:
            # Already placed in a higher-priority tier: never duplicated, but the
            # additional research reason stays visible as a flag.
            if label not in player_flags[player_id]:
                player_flags[player_id].append(label)
            return
        player_flags[player_id] = [label]
        tiers[str(tier)].append(
            {
                "player_id": player_id,
                "name": _player_label(row),
                "team": row.get("team_short_name") or row.get("team_name") or "TEAM",
                "position": row.get("position_short_name") or row.get("singular_name_short") or "POS",
                "source": label,
                # Shared list object: additional research reasons appended to
                # player_flags after placement also appear on the entry.
                "flags": player_flags[player_id],
            }
        )

    for row in squad:
        add_to_tier(1, row, "squad")

    captaincy_ids = _plan_player_ids(plan, "captaincy_candidates")
    captaincy = _resolve_plan_players(conn, captaincy_ids, warnings, "captaincy_candidates")
    if captaincy:
        for row in captaincy:
            add_to_tier(2, row, "captaincy")
    else:
        manual_gaps.append("CAPTAINCY CANDIDATES: MANUAL INPUT REQUIRED — set captaincy_candidates in config/scout_plan.json")

    transfer_ids = _plan_player_ids(plan, "transfer_candidates")
    transfer_manual = _resolve_plan_players(conn, transfer_ids, warnings, "transfer_candidates")
    buy_watch = [row for row in watchlist if str(row.get("status")) == "BUY"]
    if transfer_manual:
        for row in transfer_manual:
            add_to_tier(3, row, "transfer_manual")
    for row in buy_watch:
        add_to_tier(3, row, "watchlist_buy")
    if not transfer_manual and not buy_watch:
        manual_gaps.append("TRANSFER ALTERNATIVES: none configured — add transfer_candidates or BUY watchlist entries")

    for row in watchlist:
        add_to_tier(4, row, f"watchlist:{row.get('status')}")

    for entry in plan.get("extra_players") or []:
        if not isinstance(entry, dict):
            continue
        try:
            player_id = int(entry.get("player_id"))
        except (TypeError, ValueError):
            warnings.append(f"extra_players entry without valid player_id ignored: {entry!r}")
            continue
        row = _player_row(conn, player_id)
        if row is None:
            warnings.append(f"extra_players: player_id {player_id} not found among active players — ignored")
            continue
        tier = entry.get("tier", 5)
        try:
            tier = int(tier)
        except (TypeError, ValueError):
            tier = 5
        tier = tier if tier in TIER_CAPS else 5
        add_to_tier(tier, row, "trigger")
        question = entry.get("question")
        if isinstance(question, str) and question.strip():
            # Recorded in a local derived structure; the loaded plan object is
            # treated as immutable and never written back to.
            extra_questions.setdefault(player_id, []).append(question.strip())

    for tier, cap in TIER_CAPS.items():
        entries = tiers[str(tier)]
        if len(entries) > cap:
            dropped = [entry["name"] for entry in entries[cap:]]
            warnings.append(f"Tier {tier} capped at {cap} players; dropped: {', '.join(dropped)}")
            tiers[str(tier)] = entries[:cap]
    _apply_global_player_cap(tiers, warnings)
    universe_ids = [entry["player_id"] for entries in tiers.values() for entry in entries]

    plan_custom = {
        int(player_id): [question for question in questions if isinstance(question, str)]
        for player_id, questions in (plan.get("custom_questions") or {}).items()
        if str(player_id).isdigit()
    }
    custom_by_id: dict[int, list[str]] = {}
    for player_id in set(plan_custom) | set(extra_questions):
        custom_by_id[player_id] = list(plan_custom.get(player_id, [])) + list(extra_questions.get(player_id, []))
    watch_reasons = {
        int(row["player_id"]): [str(row["reason"])]
        for row in watchlist
        if row.get("reason")
    }
    current_rows = repo.scouting_current_rows(conn, universe_ids)
    notes_by_player: dict[int, list[dict[str, Any]]] = {}
    for row in current_rows:
        notes_by_player.setdefault(int(row["player_id"]), []).append(row)
    for notes in notes_by_player.values():
        notes.sort(key=lambda note: str(note.get("key")))

    universe_entries = [entry for entries in tiers.values() for entry in entries]
    questions_by_player: dict[int, list[str]] = {}
    for entry in universe_entries:
        player_id = entry["player_id"]
        notes = notes_by_player.get(player_id, [])
        questions_by_player[player_id] = open_questions_for_player(
            notes,
            now_dt,
            deadline_dt,
            stale_days,
            custom_by_id.get(player_id),
            signals=custom_by_id.get(player_id, []) + watch_reasons.get(player_id, []),
        )

    current_notes = [
        {
            "player_id": player_id,
            "name": next(entry["name"] for entry in universe_entries if entry["player_id"] == player_id),
            "notes": [
                {
                    "key": note.get("key"),
                    "value": _note_text(note),
                    "confidence": note.get("confidence"),
                    "observed_at": note.get("observed_at"),
                    "expires_at": note.get("expires_at"),
                    "status": _note_status(note, now_dt, deadline_dt, stale_days),
                    "observation": note.get("observation"),
                }
                for note in notes
            ],
        }
        for player_id, notes in sorted(notes_by_player.items())
        if player_id in questions_by_player
    ]

    return {
        "title": f"LUNA SCOUT BRIEF — GW{event_id}",
        "generated": now_str,
        "gameweek": event_id,
        "deadline": deadline,
        "protocol": PROTOCOL_FILENAME,
        "instructions": list(CONDENSED_INSTRUCTIONS),
        "manual_input_gaps": manual_gaps,
        "warnings": warnings,
        "squad": [
            {
                "player_id": int(row["player_id"]),
                "name": _player_label(row),
                "team": row.get("team_name") or row.get("team_short_name") or "TEAM",
                "position": row.get("singular_name_short") or "POS",
            }
            for row in squad
        ],
        "watchlist": [
            {
                "player_id": int(row["player_id"]),
                "name": _player_label(row),
                "team": row.get("team_short_name") or row.get("team_name") or "TEAM",
                "position": row.get("position_short_name") or "POS",
                "status": row.get("status"),
                "reason": row.get("reason"),
                "target_event_from": row.get("target_event_from"),
                "target_event_to": row.get("target_event_to"),
            }
            for row in watchlist
        ],
        "tiers": tiers,
        "questions_by_player": {str(player_id): questions for player_id, questions in questions_by_player.items()},
        "current_notes": current_notes,
        "status_counts": _status_counts(current_notes),
    }


def _status_counts(current_notes: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"notes": 0, "current": 0, "stale": 0, "expiring": 0, "contradictions": 0, "unresolved": 0}
    for player in current_notes:
        for note in player["notes"]:
            counts["notes"] += 1
            if note["status"] == "STALE":
                counts["stale"] += 1
            elif note["status"] == "expires before deadline":
                counts["expiring"] += 1
            else:
                counts["current"] += 1
            # Markers are scanned in every note's value and observation text,
            # not only tactical_note.
            text = f"{note.get('value') or ''} {note.get('observation') or ''}"
            if CONTRADICTION_TOKEN in text:
                counts["contradictions"] += 1
            if UNRESOLVED_TOKEN in text:
                counts["unresolved"] += 1
    return counts


def _deadline_suffix(brief: dict[str, Any]) -> str:
    deadline_dt = parse_utc(brief.get("deadline"))
    generated_dt = parse_utc(brief.get("generated"))
    if deadline_dt and generated_dt:
        hours = (deadline_dt - generated_dt).total_seconds() / 3600
        if hours >= 0:
            return f" (in {hours:.0f}h)"
        return f" ({-hours:.0f}h ago)"
    return ""


def render_scout_job(brief: dict[str, Any]) -> str:
    lines = [
        "SCOUT JOB",
        f"Gameweek: {brief['gameweek']}",
        f"Deadline: {brief['deadline'] or 'unavailable'}{_deadline_suffix(brief)}",
        f"Protocol: {brief['protocol']} (sections A1-A17, prompts B-D, QC checklist F)",
        "",
        "Research these players:",
    ]
    tier_titles = {
        "1": "TIER 1 — squad (open questions; cheap re-affirmation where fresh)",
        "2": "TIER 2 — captaincy candidates (highest evidence bar)",
        "3": "TIER 3 — live transfer alternatives",
        "4": "TIER 4 — watchlist / future targets (light touch)",
        "5": "TIER 5 — trigger-driven situations",
    }
    for tier, entries in brief["tiers"].items():
        if not entries:
            continue
        lines.append("")
        lines.append(tier_titles.get(tier, f"TIER {tier}"))
        for entry in entries:
            questions = brief["questions_by_player"].get(str(entry["player_id"]), [])
            lines.append(f"Player ID {entry['player_id']} — {entry['name']} — {entry['team']} — {entry['position']}")
            if len(entry.get("flags", [])) > 1:
                lines.append("Flags:")
                lines.extend(f"- {flag}" for flag in entry["flags"])
            lines.append("Questions:")
            lines.extend(f"- {question}" for question in questions)
    lines.append("")
    lines.append("OUTPUT:")
    lines.append("Valid FPL Brain scouting JSON (schema_version 1.0, canonical observations, controlled keys,")
    lines.append("evidence arrays, observed_at/expires_at) plus a plain-text research log.")
    return "\n".join(lines)


def render_brief_markdown(brief: dict[str, Any]) -> str:
    lines = [
        f"# {brief['title']}",
        "",
        f"Generated: {brief['generated']} | Deadline: {brief['deadline'] or 'unavailable'}{_deadline_suffix(brief)} | "
        f"Protocol: {brief['protocol']}",
        "",
        "## EXECUTION RULES (condensed — full rules in the protocol)",
        "",
        "```text",
        *brief["instructions"],
        "```",
    ]
    lines.extend(["", "## MANUAL INPUT GAPS", ""])
    if brief["manual_input_gaps"]:
        lines.extend(f"- {gap}" for gap in brief["manual_input_gaps"])
    else:
        lines.append("- none")
    if brief["warnings"]:
        lines.extend(["", "## WARNINGS", "", *(f"- {warning}" for warning in brief["warnings"])])
    lines.extend(["", "## RESEARCH UNIVERSE (agent-ready SCOUT JOB)", "", "```text"])
    lines.append(render_scout_job(brief))
    lines.append("```")
    lines.extend(["", "## CURRENT SCOUTING NOTES", ""])
    if not brief["current_notes"]:
        lines.append("No current scouting notes for the research universe.")
    for player in brief["current_notes"]:
        lines.append(f"### {player['name']} (ID {player['player_id']})")
        for note in player["notes"]:
            expires = f", expires {note['expires_at']}" if note.get("expires_at") else ""
            lines.append(
                f"- {note['key']}: {note['value']} (confidence {note['confidence']}, "
                f"observed {note['observed_at']}{expires}) [{note['status']}]"
            )
            if note.get("observation"):
                lines.append(f"  Note: {note['observation']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def scout_status(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    gw: int | None = None,
    plan: dict[str, Any] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Operational diagnostic of scouting coverage.  Not a player recommendation."""

    brief = build_brief(conn, config, gw, plan, now)
    universe = [entry for entries in brief["tiers"].values() for entry in entries]
    needing = [
        (entry, questions)
        for entry in universe
        for questions in [brief["questions_by_player"].get(str(entry["player_id"]), [])]
        if _is_research_needed(questions)
    ]
    needing.sort(key=lambda item: (-len(item[1]), item[0]["name"]))
    priority = [
        {
            "player_id": entry["player_id"],
            "name": entry["name"],
            "questions": questions,
        }
        for entry, questions in needing
    ]
    import_row = conn.execute(
        "SELECT agent, imported_at, players_total, players_resolved, notes_inserted FROM scouting_imports ORDER BY id DESC LIMIT 1"
    ).fetchone()
    fetch_run = repo.latest_fetch_run(conn, "fetch_fpl")
    return {
        "gameweek": brief["gameweek"],
        "deadline": brief["deadline"],
        "squad_players": len(brief["squad"]),
        "active_watchlist": len(brief["watchlist"]),
        "universe_players": len(universe),
        "counts": brief["status_counts"],
        "players_needing_research": len(priority),
        "priority": priority,
        "manual_input_gaps": brief["manual_input_gaps"],
        "warnings": brief["warnings"],
        "latest_scouting_import": dict(import_row) if import_row else None,
        "latest_fetch": (
            {"id": fetch_run["id"], "status": fetch_run["status"], "finished_at": fetch_run["finished_at"]}
            if fetch_run
            else None
        ),
    }


def render_status_text(status: dict[str, Any]) -> str:
    counts = status["counts"]
    lines = [
        f"FPL SCOUT STATUS — GW{status['gameweek']}",
        f"Deadline: {status['deadline'] or 'unavailable'}",
        "",
        f"Squad players: {status['squad_players']}",
        f"Active watchlist: {status['active_watchlist']}",
        f"Research universe: {status['universe_players']}",
        "",
        f"Current scouting notes: {counts['notes']}",
        f"Current: {counts['current']} | Expiring before deadline: {counts['expiring']} | Stale: {counts['stale']}",
        f"Contradictions: {counts['contradictions']} | Unresolved: {counts['unresolved']}",
        "",
        f"Players needing research: {status['players_needing_research']}",
        "",
        "Priority (most open questions first — diagnostic, not a recommendation):",
    ]
    if not status["priority"]:
        lines.append("  none — coverage is current")
    for index, item in enumerate(status["priority"], start=1):
        summary = "; ".join(item["questions"][:2])
        more = f" (+{len(item['questions']) - 2} more)" if len(item["questions"]) > 2 else ""
        lines.append(f"  {index}. {item['name']} (ID {item['player_id']}) — {summary}{more}")
    if status["manual_input_gaps"]:
        lines.extend(["", "Manual input gaps:"])
        lines.extend(f"  - {gap}" for gap in status["manual_input_gaps"])
    if status["warnings"]:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in status["warnings"])
    latest_import = status["latest_scouting_import"]
    lines.extend(["", "Latest scouting import:"])
    if latest_import:
        lines.append(
            f"  agent: {latest_import.get('agent') or 'unknown'} | time: {latest_import.get('imported_at')} | "
            f"players {latest_import.get('players_resolved')}/{latest_import.get('players_total')} | "
            f"notes {latest_import.get('notes_inserted')}"
        )
    else:
        lines.append("  none yet")
    latest_fetch = status["latest_fetch"]
    lines.extend(["", "Latest FPL fetch:"])
    if latest_fetch:
        lines.append(f"  run {latest_fetch['id']}: status {latest_fetch['status']} at {latest_fetch['finished_at']}")
    else:
        lines.append("  never")
    return "\n".join(lines)
