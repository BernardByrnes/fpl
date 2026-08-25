from __future__ import annotations

import copy
import json

from fpl_brain import repositories as repo
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database
from fpl_brain.models import EventRecord, PickRecord, PlayerRecord, PositionRecord, TeamRecord
from fpl_brain.scout_ops import build_brief, load_scout_plan, render_brief_markdown, render_status_text, scout_status
from fpl_brain.scouting import import_scouting

NOW = "2026-08-19T18:00:00Z"
DEADLINE = "2026-08-21T17:30:00Z"

PLAYERS = [
    (1, "Raya", 1, 1),
    (2, "Salah", 2, 3),
    (3, "Gabriel", 1, 2),
    (4, "Szoboszlai", 2, 3),
    (5, "Timber", 1, 2),
    (6, "Diaz", 2, 4),
    (7, "Haaland", 2, 4),
]

NOTES_DOCUMENT = {
    "schema_version": "1.0",
    "generated_at": "2026-08-18T12:00:00Z",
    "agent": "test-scout",
    "gameweek": 1,
    "default_expires_at": "2026-08-25T17:30:00Z",
    "players": [
        {
            "player_id": 1,
            "player_name": "Raya",
            "observations": [
                {"key": "start_probability", "value": 85, "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z", "observation": "First choice keeper."},
                {"key": "expected_minutes", "value": 80, "confidence": "medium", "observed_at": "2026-07-30T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                {"key": "penalty_probability", "value": 50, "confidence": "medium", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
            ],
        },
        {
            "player_id": 2,
            "player_name": "Salah",
            "observations": [
                {"key": "start_probability", "value": 70, "confidence": "medium", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-20T17:30:00Z"},
                {"key": "tactical_note", "value": "CONTRADICTION: presser says fit, reporter says doubt", "confidence": "low", "observed_at": "2026-08-18T12:00:00Z"},
            ],
        },
        {
            "player_id": 3,
            "player_name": "Gabriel",
            "observations": [
                {"key": "injury_uncertainty", "value": 70, "confidence": "low", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
            ],
        },
        {
            "player_id": 5,
            "player_name": "Timber",
            "observations": [
                {"key": "tactical_note", "value": "UNRESOLVED: role after new signing unknown", "confidence": "low", "observed_at": "2026-08-18T12:00:00Z"},
            ],
        },
        {
            "player_id": 7,
            "player_name": "Haaland",
            "observations": [
                {"key": "start_probability", "value": 95, "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                {"key": "expected_minutes", "value": 90, "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                {"key": "likely_role", "value": "number nine", "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                {"key": "role_security_5gw", "value": "high", "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                {"key": "penalty_probability", "value": 85, "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                {"key": "rotation_risk", "value": "low", "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
            ],
        },
    ],
}


def _seed(tmp_path, notes=True):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["fpl_entry_id"] = 99
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="Arsenal", short_name="ARS"), TeamRecord(id=2, name="Liverpool", short_name="LIV")])
        repo.upsert_positions(
            conn,
            [
                PositionRecord(id=1, singular_name_short="GKP"),
                PositionRecord(id=2, singular_name_short="DEF"),
                PositionRecord(id=3, singular_name_short="MID"),
                PositionRecord(id=4, singular_name_short="FWD"),
            ],
        )
        repo.upsert_players(
            conn,
            [PlayerRecord(id=pid, web_name=name, full_name=name, norm_name=name.lower(), team_id=team, element_type=pos) for pid, name, team, pos in PLAYERS],
        )
        repo.upsert_events(conn, [EventRecord(id=1, name="Gameweek 1", deadline_time=DEADLINE, is_next=1)])
        repo.upsert_squad_picks(
            conn,
            99,
            1,
            [
                PickRecord(player_id=1, position=1, multiplier=1),
                PickRecord(player_id=2, position=2, multiplier=1),
                PickRecord(player_id=7, position=3, multiplier=1),
            ],
        )
        repo.add_watchlist(conn, 4, "BUY", reason="mid-price target")
        repo.add_watchlist(conn, 3, "WATCH", reason="monitor")
        repo.add_watchlist(conn, 5, "WATCH", reason="monitor")
    conn.close()
    if notes:
        source = tmp_path / "notes.json"
        source.write_text(json.dumps(NOTES_DOCUMENT), encoding="utf-8")
        result = import_scouting(config, source)
        assert result["players_resolved"] == 5
    return config


def _connect(config):
    conn = connect_database(config["paths"]["database"])
    return conn


PLAN = {
    "gameweek": 1,
    "captaincy_candidates": [3, 9999],
    "transfer_candidates": [6],
    "extra_players": [],
    "custom_questions": {"1": ["Confirm penalty hierarchy"]},
}


def test_brief_tiers_combine_squad_watchlist_and_manual_plan(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, PLAN, NOW)
    finally:
        conn.close()
    assert [entry["player_id"] for entry in brief["tiers"]["1"]] == [1, 2, 7]
    assert [entry["player_id"] for entry in brief["tiers"]["2"]] == [3]
    assert [entry["player_id"] for entry in brief["tiers"]["3"]] == [6, 4]
    assert [entry["player_id"] for entry in brief["tiers"]["4"]] == [5]
    assert brief["tiers"]["5"] == []
    assert any("player_id 9999" in warning for warning in brief["warnings"])
    assert any(question == "Confirm penalty hierarchy" for question in brief["questions_by_player"]["1"])


def test_brief_reports_manual_input_gaps_when_plan_missing(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {}, NOW)
    finally:
        conn.close()
    assert any(gap.startswith("CAPTAINCY CANDIDATES: MANUAL INPUT REQUIRED") for gap in brief["manual_input_gaps"])
    # A BUY watchlist entry satisfies the transfer tier without manual input.
    assert not any(gap.startswith("TRANSFER ALTERNATIVES") for gap in brief["manual_input_gaps"])


def test_stale_and_expiring_note_detection(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {}, NOW)
    finally:
        conn.close()
    assert "expected_minutes is STALE (observed 2026-07-30T12:00:00Z) — re-affirm or refresh" in brief["questions_by_player"]["1"]
    assert any("expires 2026-08-20T17:30:00Z" in question for question in brief["questions_by_player"]["2"])
    statuses = {note["key"]: note["status"] for player in brief["current_notes"] if player["player_id"] == 1 for note in player["notes"]}
    assert statuses["expected_minutes"] == "STALE"
    assert statuses["start_probability"] == "current"


def test_contradiction_and_unresolved_detection(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {}, NOW)
    finally:
        conn.close()
    assert any("CONTRADICTION" in question for question in brief["questions_by_player"]["2"])
    assert any("UNRESOLVED" in question for question in brief["questions_by_player"]["5"])
    assert brief["status_counts"]["contradictions"] == 1
    assert brief["status_counts"]["unresolved"] == 1


def test_missing_notes_become_open_questions(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {}, NOW)
    finally:
        conn.close()
    assert brief["questions_by_player"]["4"] == ["no scouting notes at all — establish minutes, role, and set-piece view"]
    # No set-piece signal exists for player 3, so absence of set-piece notes
    # must NOT produce a set-piece research question.
    assert not any("set-piece" in question for question in brief["questions_by_player"]["3"])


def test_complete_player_gets_reaffirmation_only(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {}, NOW)
    finally:
        conn.close()
    assert brief["questions_by_player"]["7"] == ["no open question — cheap re-affirmation only (protocol A1)"]


def test_load_scout_plan_tolerates_missing_and_invalid_files(tmp_path):
    plan, warnings = load_scout_plan(tmp_path / "missing.json")
    assert plan == {} and warnings
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    plan, warnings = load_scout_plan(bad)
    assert plan == {} and warnings
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"captaincy_candidates": [2]}), encoding="utf-8")
    plan, warnings = load_scout_plan(good)
    assert plan == {"captaincy_candidates": [2]} and not warnings


def test_protocol_reference_present_and_no_recommendations(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, PLAN, NOW)
        status = scout_status(conn, config, 1, PLAN, NOW)
    finally:
        conn.close()
    markdown = render_brief_markdown(brief)
    assert "LUNA_SCOUT_PROTOCOL_V1.md" in markdown
    assert brief["protocol"] == "LUNA_SCOUT_PROTOCOL_V1.md"
    assert "diagnostic, not a recommendation" in render_status_text(status)
    for line in markdown.splitlines():
        stripped = line.lstrip("- ").lstrip()
        assert not stripped.lower().startswith(("buy ", "sell ", "captain ", "wildcard "))


def test_deterministic_output_with_fixed_now(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        first = build_brief(conn, config, 1, PLAN, NOW)
        second = build_brief(conn, config, 1, PLAN, NOW)
    finally:
        conn.close()
    assert render_brief_markdown(first) == render_brief_markdown(second)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_status_counts_and_priority(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        status = scout_status(conn, config, 1, {}, NOW)
    finally:
        conn.close()
    assert status["squad_players"] == 3
    assert status["active_watchlist"] == 3
    assert status["counts"]["notes"] == 13
    assert status["counts"]["expiring"] == 1
    assert status["counts"]["stale"] == 1
    assert status["counts"]["contradictions"] == 1
    assert status["counts"]["unresolved"] == 1
    assert status["players_needing_research"] == 5  # squad 1,2 + watchlist 3,4,5; player 7 is fully covered
    assert status["universe_players"] == 6  # player 6 needs the manual plan to enter the universe
    assert status["priority"][0]["name"] == "Salah"
    assert status["latest_scouting_import"]["agent"] == "test-scout"


def _import_notes(config, tmp_path, document, name):
    source = tmp_path / name
    source.write_text(json.dumps(document), encoding="utf-8")
    result = import_scouting(config, source)
    assert result["unresolved"] == []


def _seed_wide(tmp_path, player_count=44):
    """Adversarial database: squad 15, captaincy 5, transfer 8, watchlist 8, triggers 8."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["fpl_entry_id"] = 99
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="Arsenal", short_name="ARS")])
        repo.upsert_positions(conn, [PositionRecord(id=3, singular_name_short="MID")])
        repo.upsert_players(
            conn,
            [
                PlayerRecord(id=pid, web_name=f"P{pid:02d}", full_name=f"Player {pid:02d}", norm_name=f"player {pid:02d}", team_id=1, element_type=3)
                for pid in range(1, player_count + 1)
            ],
        )
        repo.upsert_events(conn, [EventRecord(id=1, name="Gameweek 1", deadline_time=DEADLINE, is_next=1)])
        repo.upsert_squad_picks(
            conn,
            99,
            1,
            [PickRecord(player_id=pid, position=slot, multiplier=1) for slot, pid in enumerate(range(1, 16), start=1)],
        )
        for pid in range(29, 37):
            repo.add_watchlist(conn, pid, "WATCH", reason="monitor")
    conn.close()
    plan = {
        "captaincy_candidates": list(range(16, 21)),
        "transfer_candidates": list(range(21, 29)),
        "extra_players": [
            {"player_id": pid, "tier": 5, "question": f"Trigger situation for P{pid:02d}"} for pid in range(37, 45)
        ],
        "custom_questions": {},
    }
    return config, plan


def test_global_cap_adversarial_44_candidates_trimmed_to_40(tmp_path):
    config, plan = _seed_wide(tmp_path)
    conn = _connect(config)
    try:
        first = build_brief(conn, config, 1, plan, NOW)
        second = build_brief(conn, config, 1, plan, NOW)
    finally:
        conn.close()
    sizes = {tier: len(entries) for tier, entries in first["tiers"].items()}
    assert sizes == {"1": 15, "2": 5, "3": 8, "4": 8, "5": 4}
    assert sum(sizes.values()) == 40
    # Deterministic: trim from the end of the lowest tier, same four omitted.
    assert [entry["player_id"] for entry in first["tiers"]["5"]] == [37, 38, 39, 40]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    cap_warnings = [warning for warning in first["warnings"] if warning.startswith("GLOBAL CAP APPLIED:")]
    assert len(cap_warnings) == 1
    for omitted_id in (41, 42, 43, 44):
        assert f"ID {omitted_id}" in cap_warnings[0]
    # Questions, notes, and markdown all reflect the same post-cap universe.
    for omitted_id in (41, 42, 43, 44):
        assert str(omitted_id) not in first["questions_by_player"]
        assert all(player["player_id"] != omitted_id for player in first["current_notes"])
    markdown = render_brief_markdown(first)
    assert "GLOBAL CAP APPLIED" in markdown
    assert "Player ID 44" not in markdown
    status_conn = _connect(config)
    try:
        status = scout_status(status_conn, config, 1, plan, NOW)
    finally:
        status_conn.close()
    assert status["universe_players"] == 40


def test_normal_universe_below_cap_is_never_trimmed(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, PLAN, NOW)
    finally:
        conn.close()
    assert not any(warning.startswith("GLOBAL CAP APPLIED") for warning in brief["warnings"])
    assert sum(len(entries) for entries in brief["tiers"].values()) == len(
        {entry["player_id"] for entries in brief["tiers"].values() for entry in entries}
    )


def test_repeat_build_does_not_mutate_manual_plan(tmp_path):
    config = _seed(tmp_path)
    plan = {
        "extra_players": [{"player_id": 6, "tier": 5, "question": "Check role after new signing"}],
        "custom_questions": {"1": ["Confirm penalty hierarchy"]},
    }
    snapshot = copy.deepcopy(plan)
    conn = _connect(config)
    try:
        first = build_brief(conn, config, 1, plan, NOW)
        second = build_brief(conn, config, 1, plan, NOW)
    finally:
        conn.close()
    assert plan == snapshot
    for brief in (first, second):
        assert brief["questions_by_player"]["6"].count("Check role after new signing") == 1
        assert brief["questions_by_player"]["1"].count("Confirm penalty hierarchy") == 1


def test_captaincy_overlap_stays_tier_1_with_visible_flag(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {"captaincy_candidates": [1]}, NOW)
    finally:
        conn.close()
    occurrences = [entry for entries in brief["tiers"].values() for entry in entries if entry["player_id"] == 1]
    assert len(occurrences) == 1
    assert brief["tiers"]["2"] == []
    assert occurrences[0]["flags"] == ["CURRENT SQUAD", "CAPTAINCY CANDIDATE"]
    markdown = render_brief_markdown(brief)
    assert "Flags:" in markdown
    assert "- CAPTAINCY CANDIDATE" in markdown
    assert "- CURRENT SQUAD" in markdown


def test_contradiction_and_unresolved_markers_detected_in_any_note(tmp_path):
    config = _seed(tmp_path)
    _import_notes(
        config,
        tmp_path,
        {
            "schema_version": "1.0",
            "generated_at": "2026-08-18T12:00:00Z",
            "agent": "test-scout",
            "gameweek": 1,
            "default_expires_at": "2026-08-25T17:30:00Z",
            "players": [
                {
                    "player_id": 4,
                    "player_name": "Szoboszlai",
                    "observations": [
                        {"key": "penalty_probability", "value": 50, "confidence": "low", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z", "observation": "CONTRADICTION: club says taker A, reporter says taker B."},
                    ],
                },
                {
                    "player_id": 6,
                    "player_name": "Diaz",
                    "observations": [
                        {"key": "likely_role", "value": "wide forward", "confidence": "low", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z", "observation": "UNRESOLVED: role after the system change."},
                    ],
                },
            ],
        },
        "markers.json",
    )
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, PLAN, NOW)  # PLAN puts 4 and 6 in tier 3
    finally:
        conn.close()
    assert any("CONTRADICTION" in question for question in brief["questions_by_player"]["4"])
    assert any("UNRESOLVED" in question for question in brief["questions_by_player"]["6"])
    assert brief["status_counts"]["contradictions"] == 2  # tactical note on 2 + penalty note on 4
    assert brief["status_counts"]["unresolved"] == 2  # tactical note on 5 + likely_role note on 6


def test_custom_injury_question_surfaces_missing_injury_uncertainty(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {"custom_questions": {"1": ["Is he over his injury?"]}}, NOW)
    finally:
        conn.close()
    assert any(
        question.startswith("injury_uncertainty missing despite an explicit injury/fitness question")
        for question in brief["questions_by_player"]["1"]
    )


def test_custom_set_piece_question_identifies_partial_coverage(tmp_path):
    config = _seed(tmp_path)  # player 1 has penalty_probability only
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {"custom_questions": {"1": ["Who takes penalties and corners?"]}}, NOW)
    finally:
        conn.close()
    coverage = [q for q in brief["questions_by_player"]["1"] if q.startswith("set-piece coverage incomplete")]
    assert len(coverage) == 1
    assert "freekick_probability" in coverage[0] and "corner_probability" in coverage[0] and "set_piece_role" in coverage[0]
    # The fresh, already-covered penalty_probability must not be listed as missing.
    assert "penalty_probability" not in coverage[0]


SET_PIECE_QUESTION_MARKERS = ("penalt", "set-piece", "free kick", "freekick", "corner")


def _import_minutes_notes(config, tmp_path, player_id, player_name):
    _import_notes(
        config,
        tmp_path,
        {
            "schema_version": "1.0",
            "generated_at": "2026-08-18T12:00:00Z",
            "agent": "test-scout",
            "gameweek": 1,
            "default_expires_at": "2026-08-25T17:30:00Z",
            "players": [
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "observations": [
                        {"key": "start_probability", "value": 85, "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                        {"key": "expected_minutes", "value": 80, "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                        {"key": "likely_role", "value": "left eight", "confidence": "high", "observed_at": "2026-08-18T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                    ],
                }
            ],
        },
        f"minutes_{player_id}.json",
    )


def test_set_piece_absence_without_signal_produces_no_set_piece_question(tmp_path):
    config = _seed(tmp_path)
    _import_minutes_notes(config, tmp_path, 6, "Diaz")  # PLAN puts player 6 in tier 3
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, PLAN, NOW)
    finally:
        conn.close()
    questions = brief["questions_by_player"]["6"]
    assert questions  # minutes/role questions may exist...
    for question in questions:
        assert not any(marker in question.lower() for marker in SET_PIECE_QUESTION_MARKERS)


def test_explicit_set_piece_signal_with_zero_coverage_produces_one_question(tmp_path):
    config = _seed(tmp_path)
    _import_minutes_notes(config, tmp_path, 6, "Diaz")
    conn = _connect(config)
    try:
        brief = build_brief(
            conn,
            config,
            1,
            {"transfer_candidates": [6], "custom_questions": {"6": ["Check set-piece hierarchy"]}},
            NOW,
        )
    finally:
        conn.close()
    questions = brief["questions_by_player"]["6"]
    generated = [question for question in questions if question.startswith("set-piece hierarchy unknown")]
    assert generated == [
        "set-piece hierarchy unknown — set-piece research is requested but no penalty/free-kick/corner/set_piece_role note exists"
    ]
    # The manual question itself is echoed verbatim exactly once and is the only other set-piece mention.
    assert questions.count("Check set-piece hierarchy") == 1
    assert len([q for q in questions if any(marker in q.lower() for marker in SET_PIECE_QUESTION_MARKERS)]) == 2


def test_stale_set_piece_note_produces_refresh_question_only(tmp_path):
    config = _seed(tmp_path)
    _import_notes(
        config,
        tmp_path,
        {
            "schema_version": "1.0",
            "generated_at": "2026-08-18T12:00:00Z",
            "agent": "test-scout",
            "gameweek": 1,
            "default_expires_at": "2026-08-25T17:30:00Z",
            "players": [
                {
                    "player_id": 4,
                    "player_name": "Szoboszlai",
                    "observations": [
                        {"key": "penalty_probability", "value": 50, "confidence": "medium", "observed_at": "2026-07-30T12:00:00Z", "expires_at": "2026-08-25T17:30:00Z"},
                    ],
                }
            ],
        },
        "stale_penalty.json",
    )
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {}, NOW)  # player 4 is a BUY watchlist entry, no set-piece signal
    finally:
        conn.close()
    questions = brief["questions_by_player"]["4"]
    assert any(question.startswith("penalty_probability is STALE (observed 2026-07-30T12:00:00Z)") for question in questions)
    # Without an explicit set-piece signal, no missing-coverage question is inferred.
    assert not any(question.startswith(("set-piece hierarchy unknown", "set-piece coverage incomplete")) for question in questions)


def test_no_transfer_signal_means_no_transfer_question(tmp_path):
    config = _seed(tmp_path)
    conn = _connect(config)
    try:
        brief = build_brief(conn, config, 1, {}, NOW)
    finally:
        conn.close()
    for questions in brief["questions_by_player"].values():
        assert not any("transfer_exit_risk" in question for question in questions)
