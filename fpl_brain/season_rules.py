"""Season rules representation derived from official game settings.

The rule constants the decision-support code depends on (free-transfer cap,
hit cost, selling-price engine parameters) live in one struct with explicit
transitions, so the official 2026/27 values are read from the official
bootstrap `game_settings` payload instead of being hardcoded everywhere.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import raw_archive

REQUIRED_GAME_SETTINGS_KEYS = (
    "squad_squadsize",
    "squad_squadplay",
    "squad_team_limit",
    "squad_total_spend",
    "max_extra_free_transfers",
    "transfers_sell_on_fee",
    "element_sell_at_purchase_price",
)

BASE_FREE_TRANSFERS = 1
SEASON_TRANSFER_HIT_COST = 4  # official FPL constant; not published inside game_settings

#: Prefix for the contiguous-window provenance flag.  The flag is ALWAYS
#: derived from the real decision events so an artifact can never claim a window
#: it did not run.  This is the single definition used by every producer: two
#: independent copies of this logic had diverged before (one looked the first
#: event up by NUMBER, which is silently zero for any window not containing it).
WINDOW_FLAG_PREFIX = "SUPPORTED_CONTIGUOUS_EVENTS"


def window_flag(events) -> str:
    """Truthful window label, e.g. ``(5, 6, 7, 8)`` -> ``SUPPORTED_CONTIGUOUS_EVENTS_5_6_7_8``."""

    return WINDOW_FLAG_PREFIX + "".join(f"_{int(event)}" for event in events)


class SeasonRulesError(ValueError):
    """Official game settings are missing or contradict required rule constants."""


PINNED_RULES_EVIDENCE_SCHEMA = "fpl_brain.origin_pinned_season_rules.v1"


@dataclass(frozen=True)
class PinnedSeasonRules:
    rules: "SeasonRules"
    evidence: Mapping[str, Any]


def _utc(value: Any, *, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as failure:
        raise SeasonRulesError(f"{name} is not an ISO timestamp") from failure
    if parsed.tzinfo is None:
        raise SeasonRulesError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_origin_pinned_season_rules(
    conn: sqlite3.Connection,
    *,
    season: str,
    cutoff: str,
    data_snapshot_sha256: str,
) -> PinnedSeasonRules:
    """Resolve season rules from accepted official settings in a verified snapshot.

    The connection must be the read-only database snapshot whose content digest
    is ``data_snapshot_sha256``. The selected accepted bootstrap generation pins
    a fetch run in that snapshot; its immutable bootstrap archive capture is
    re-hashed before deriving rules. Historical rows without that retained
    capture remain unavailable; they are never reconstructed from today's
    defaults.
    """

    origin = _utc(cutoff, name="origin cutoff")
    snapshot = str(data_snapshot_sha256 or "").removeprefix("sha256:")
    if len(snapshot) != 64 or any(character not in "0123456789abcdef" for character in snapshot.lower()):
        raise SeasonRulesError("origin data snapshot identity is not a SHA-256 digest")
    try:
        rows = conn.execute(
            "SELECT bg.id, bg.fetch_run_id, bg.captured_at, fr.raw_dir "
            "FROM bootstrap_generations AS bg LEFT JOIN fetch_runs AS fr "
            "ON fr.id=bg.fetch_run_id WHERE bg.accepted=1 AND bg.fetch_run_id IS NOT NULL "
            "ORDER BY bg.id DESC",
        ).fetchall()
    except sqlite3.OperationalError as failure:
        raise SeasonRulesError("pinned snapshot has no accepted bootstrap provenance") from failure

    candidates = []
    for row in rows:
        captured_at = _utc(row["captured_at"], name="bootstrap captured_at")
        if captured_at <= origin:
            candidates.append((captured_at, int(row["id"]), row))
    if not candidates:
        raise SeasonRulesError("pinned snapshot has no accepted bootstrap capture at or before the cutoff")
    captured_at, generation_id, row = max(candidates, key=lambda item: (item[0], item[1]))
    fetch_run_id = row["fetch_run_id"]
    raw_dir = str(row["raw_dir"] or "").strip()
    if not raw_dir:
        raise SeasonRulesError("accepted bootstrap generation has no retained raw archive directory")
    try:
        captures = [
            record for record in raw_archive.load_manifest(raw_dir)
            if str(record.get("source") or "") == "bootstrap_static"
            and str(record.get("run_id") or "") == str(fetch_run_id)
            and _utc(record.get("observed_at"), name="bootstrap archive observed_at") == captured_at
        ]
    except Exception as failure:
        raise SeasonRulesError(f"pinned bootstrap archive manifest is unavailable or malformed: {failure}") from failure
    if len(captures) != 1:
        raise SeasonRulesError("accepted bootstrap generation does not resolve to exactly one archived payload")
    capture = captures[0]
    if not raw_archive.verify_archived_blob(raw_dir, capture):
        raise SeasonRulesError("pinned bootstrap archive payload digest does not verify")
    try:
        payload_path = Path(raw_dir) / raw_archive.ARCHIVE_DIRNAME / str(capture["relative_path"])
        payload = json.loads(payload_path.read_bytes())
    except (OSError, KeyError, TypeError, ValueError) as failure:
        raise SeasonRulesError("pinned bootstrap archive payload is unreadable or malformed") from failure
    if not isinstance(payload, Mapping):
        raise SeasonRulesError("pinned bootstrap archive payload is not an object")
    rules = season_rules_from_bootstrap(str(season), payload)
    settings = payload["game_settings"]
    settings_sha256 = _canonical_sha256(settings)
    rules_payload = asdict(rules)
    evidence = {
        "schema": PINNED_RULES_EVIDENCE_SCHEMA,
        "source": "ACCEPTED_BOOTSTRAP_CAPTURE_IN_ORIGIN_SNAPSHOT_ARCHIVE",
        "season": str(season),
        "bootstrap_generation_id": generation_id,
        "fetch_run_id": int(fetch_run_id),
        "captured_at": captured_at.isoformat().replace("+00:00", "Z"),
        "origin_cutoff": str(cutoff),
        "data_snapshot_sha256": snapshot,
        "archive_capture_id": str(capture.get("capture_id") or ""),
        "bootstrap_payload_sha256": str(capture.get("payload_sha256") or ""),
        "archive_relative_path": str(capture.get("relative_path") or ""),
        "game_settings_sha256": settings_sha256,
        "season_rules": rules_payload,
        "season_rules_sha256": _canonical_sha256(rules_payload),
    }
    return PinnedSeasonRules(rules=rules, evidence=evidence)


def verify_pinned_season_rules_evidence(
    evidence: Mapping[str, Any],
    rules: "SeasonRules",
    *,
    cutoff: str,
    data_snapshot_sha256: str,
    allow_fixture: bool = False,
) -> bool:
    """Check the retained rules binding against the exact opportunity origin."""

    rule_payload = asdict(rules)
    source = evidence.get("source")
    production_source = "ACCEPTED_BOOTSTRAP_CAPTURE_IN_ORIGIN_SNAPSHOT_ARCHIVE"
    fixture_source = "EXPLICIT_FIXTURE_ONLY"
    if evidence.get("schema") != PINNED_RULES_EVIDENCE_SCHEMA:
        raise SeasonRulesError("season-rule evidence schema is unsupported")
    if source != production_source and not (allow_fixture and source == fixture_source
                                            and evidence.get("evidence_class") == "FIXTURE_ONLY"):
        raise SeasonRulesError("season-rule evidence is not from a verified origin snapshot")
    if (
        evidence.get("season_rules") != rule_payload
        or evidence.get("season_rules_sha256") != _canonical_sha256(rule_payload)
        or str(evidence.get("origin_cutoff") or "") != str(cutoff)
        or str(evidence.get("data_snapshot_sha256") or "").removeprefix("sha256:")
        != str(data_snapshot_sha256 or "").removeprefix("sha256:")
        or _utc(evidence.get("captured_at"), name="season-rule captured_at")
        > _utc(cutoff, name="origin cutoff")
    ):
        raise SeasonRulesError("season-rule evidence differs from the origin rules, cutoff or data snapshot")
    if source == production_source:
        for name in ("bootstrap_generation_id", "fetch_run_id"):
            if int(evidence.get(name) or 0) <= 0:
                raise SeasonRulesError(f"season-rule evidence omits {name}")
        for name in ("archive_capture_id", "archive_relative_path"):
            if not str(evidence.get(name) or "").strip():
                raise SeasonRulesError(f"season-rule evidence omits {name}")
        for name in ("bootstrap_payload_sha256", "game_settings_sha256"):
            value = str(evidence.get(name) or "").removeprefix("sha256:")
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise SeasonRulesError(f"season-rule evidence {name} is not a SHA-256 digest")
    return True


@dataclass(frozen=True)
class SeasonRules:
    """Minimum season-scoped rule structure preventing silent rule drift."""

    season: str
    base_free_transfers: int = BASE_FREE_TRANSFERS
    max_extra_free_transfers: int = 4
    max_free_transfers: int = 5
    transfer_hit_cost: int = SEASON_TRANSFER_HIT_COST
    sell_on_fee: float = 0.5
    sell_at_purchase_price_when_falling: bool = False
    wildcard_ft_rule: str = "saved_free_transfers_preserved"
    free_hit_ft_rule: str = "saved_free_transfers_preserved"
    bench_boost_ft_rule: str = "saved_free_transfers_preserved"
    triple_captain_ft_rule: str = "saved_free_transfers_preserved"
    # Official 2026/27 Wildcard semantics, verified 2026-09-12:
    #   - every transfer made in the Gameweek is free, INCLUDING transfers
    #     already made before the Wildcard was activated (the Gameweek's hit is
    #     removed / refunded),
    #   - saved free transfers are retained,
    #   - only one chip may be active in a Gameweek,
    #   - Wildcard transfers are permanent, and the chip cannot be cancelled
    #     once confirmed.
    wildcard_covers_prior_same_gw_transfers: bool = True
    wildcard_transfers_are_permanent: bool = True
    wildcard_cancellable_once_confirmed: bool = False
    one_chip_per_gameweek: bool = True
    squad_size: int = 15
    squad_min_play: int = 11
    squad_max_play: int = 15
    squad_team_limit: int = 3
    squad_total_spend: int = 1000

    @property
    def squad_max_play_value(self) -> int:
        return self.squad_max_play


def season_rules_from_game_settings(
    season: str,
    game_settings: Mapping[str, Any] | None,
) -> SeasonRules:
    """Derive the season rules from the official bootstrap game_settings."""

    if not isinstance(game_settings, Mapping):
        raise SeasonRulesError("official game_settings payload is missing")
    missing = [key for key in REQUIRED_GAME_SETTINGS_KEYS if key not in game_settings]
    if missing:
        raise SeasonRulesError(f"official game_settings missing required keys: {', '.join(missing)}")
    extra = game_settings["max_extra_free_transfers"]
    if isinstance(extra, bool) or not isinstance(extra, int) or extra < 0:
        raise SeasonRulesError("official max_extra_free_transfers must be a non-negative integer")
    sell_on_fee = game_settings["transfers_sell_on_fee"]
    if isinstance(sell_on_fee, bool) or not isinstance(sell_on_fee, (int, float)) or not 0 <= float(sell_on_fee) <= 1:
        raise SeasonRulesError("official transfers_sell_on_fee must be a fraction between 0 and 1")
    element_sell = game_settings["element_sell_at_purchase_price"]
    if not isinstance(element_sell, bool):
        raise SeasonRulesError("official element_sell_at_purchase_price must be boolean")
    squad_size = game_settings["squad_squadsize"]
    squad_min_play = game_settings["squad_squadplay"]
    squad_team_limit = game_settings["squad_team_limit"]
    squad_total_spend = game_settings["squad_total_spend"]
    for label, value in (
        ("squad_squadsize", squad_size),
        ("squad_squadplay", squad_min_play),
        ("squad_team_limit", squad_team_limit),
        ("squad_total_spend", squad_total_spend),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SeasonRulesError(f"official {label} must be a positive integer")
    return SeasonRules(
        season=season,
        max_extra_free_transfers=extra,
        max_free_transfers=BASE_FREE_TRANSFERS + extra,
        sell_on_fee=float(sell_on_fee),
        sell_at_purchase_price_when_falling=bool(element_sell),
        squad_size=int(squad_size),
        squad_min_play=int(squad_min_play),
        squad_team_limit=int(squad_team_limit),
        squad_total_spend=int(squad_total_spend),
    )


def season_rules_from_bootstrap(season: str, payload: Mapping[str, Any]) -> SeasonRules:
    """Derive season rules from an official bootstrap-static payload."""

    if not isinstance(payload, Mapping) or "game_settings" not in payload:
        raise SeasonRulesError("bootstrap-static payload has no game_settings")
    return season_rules_from_game_settings(season, payload["game_settings"])


def free_transfers_after_gameweek(
    rules: SeasonRules,
    saved_free_transfers: int,
    free_transfers_used: int,
) -> int:
    """Free-transfer rollover for a NORMAL (no-chip) gameweek.

    Official semantics: every gameweek grants one new free transfer; unused
    banked transfers roll over up to the season cap. Using fewer than saved
    keeps the remainder banked; nothing is lost by saving.

    Do NOT use this for a Wildcard or Free Hit week — see
    :func:`free_transfers_after_chip`.
    """

    if isinstance(saved_free_transfers, int) is False or isinstance(free_transfers_used, int) is False:
        raise ValueError("free-transfer counts must be integers")
    if free_transfers_used < 0:
        raise ValueError("free_transfers_used must not be negative")
    leftover = max(0, saved_free_transfers - free_transfers_used)
    return min(rules.max_free_transfers, leftover + rules.base_free_transfers)


#: Chips that make a whole Gameweek's transfers free and RETAIN the saved FT
#: state (no weekly +1 accrual).
FT_PRESERVING_CHIPS = ("wildcard", "freehit")
#: Team chips that do not change the transfer process (normal rollover applies).
NON_TRANSFER_CHIPS = ("bboost", "3xc")

#: Every official chip name the season can present, in canonical spelling.  One
#: list, so the state layer, the route layer and the chip decision layer cannot
#: drift apart.
CANONICAL_CHIP_NAMES = ("wildcard", "freehit", "bboost", "3xc")

#: Spellings under which a chip may be requested in a payload or a route.  The
#: normaliser strips case, underscores, hyphens and spaces, so these are simply
#: the spellings that must resolve to the same four chips.
CHIP_NAME_KEYWORDS = (
    "wildcard",
    "freehit",
    "free_hit",
    "bboost",
    "benchboost",
    "bench_boost",
    "3xc",
    "triplecaptain",
    "triple_captain",
)


class ChipFreeTransferError(ValueError):
    """The event-start free-transfer bank is required but was not recorded."""


def normalise_chip_name(chip: str) -> str:
    return str(chip).lower().replace("_", "").replace("-", "").replace(" ", "")


def chip_preserves_saved_free_transfers(chip: str) -> bool:
    """True when the chip retains the saved FT state (Wildcard / Free Hit)."""

    return normalise_chip_name(chip) in FT_PRESERVING_CHIPS


def free_transfers_after_chip(
    rules: SeasonRules,
    chip: str,
    *,
    event_start_free_transfers: int | None,
    free_transfers_available: int | None = None,
    free_transfers_used: int = 0,
) -> int:
    """Free transfers entering the Gameweek AFTER a chip week.

    Official 2026/27 rule (verified 2026-09-12): a played Wildcard or Free Hit
    **retains the saved FT state** — the weekly +1 accrual does not apply, so a
    manager who held 2 saved FTs still holds 2 afterwards, regardless of how
    many chip transfers were made.

    The retained value is the bank from the START of the Gameweek, because the
    chip may be played AFTER transfers were already made in that Gameweek.  The
    event-start bank must therefore be recorded explicitly; when it is missing we
    raise instead of guessing from the possibly depleted current count.
    """

    name = normalise_chip_name(chip)
    if name in FT_PRESERVING_CHIPS:
        if event_start_free_transfers is None:
            raise ChipFreeTransferError(
                f"{chip} retains the saved free transfers but the event-start free-transfer bank "
                "was not explicitly recorded; refusing to infer it from the current remaining count"
            )
        bank = int(event_start_free_transfers)
        if bank < 0:
            raise ChipFreeTransferError("event_start_free_transfers must not be negative")
        return min(rules.max_free_transfers, bank)
    if name in NON_TRANSFER_CHIPS:
        if free_transfers_available is None:
            raise ChipFreeTransferError(f"{chip} needs the available free transfers for the rollover")
        return free_transfers_after_gameweek(rules, int(free_transfers_available), int(free_transfers_used))
    raise ChipFreeTransferError(f"unknown chip for free-transfer transition: {chip}")


def transfer_hit_cost(rules: SeasonRules, extra_transfers: int) -> int:
    """Point cost for a batch of transfers beyond the available free ones."""

    if extra_transfers < 0:
        raise ValueError("extra_transfers must not be negative")
    return extra_transfers * rules.transfer_hit_cost


def chip_keeps_saved_free_transfers(rule: str) -> bool:
    """Official chip handling: Wildcard/Free Hit/BB/TC never change saved FTs."""

    return rule == "saved_free_transfers_preserved"


def wildcard_covers_prior_same_gw_transfers(rules: SeasonRules) -> bool:
    """Official rule: a Wildcard makes ALL that Gameweek's transfers free.

    Transfers already made earlier in the same Gameweek (including any points
    hit already taken) are covered once the Wildcard is activated, so no
    transfer deduction remains for the Gameweek.
    """

    return bool(rules.wildcard_covers_prior_same_gw_transfers)


def wildcard_gameweek_hit(
    rules: SeasonRules,
    *,
    prior_paid_transfers: int = 0,
    wildcard_transfers: int = 0,
    wildcard_active: bool = True,
) -> dict[str, int | bool]:
    """Point deduction for a Gameweek in which the Wildcard is played.

    Returns the gross hit that would apply WITHOUT the chip, the amount the
    chip covers, and the net hit (0 when the chip is active and covers the
    Gameweek).  ``prior_paid_transfers`` is the count of already-charged extra
    transfers from earlier in the same Gameweek.
    """

    if prior_paid_transfers < 0 or wildcard_transfers < 0:
        raise ValueError("transfer counts must not be negative")
    gross_hit = transfer_hit_cost(rules, int(prior_paid_transfers) + int(wildcard_transfers))
    covered = gross_hit if (wildcard_active and wildcard_covers_prior_same_gw_transfers(rules)) else 0
    return {
        "wildcard_active": bool(wildcard_active),
        "gross_hit_without_chip": int(gross_hit),
        "hit_covered_by_wildcard": int(covered),
        "net_hit": int(gross_hit - covered),
        "covers_prior_same_gw_transfers": bool(wildcard_covers_prior_same_gw_transfers(rules)),
        "saved_free_transfers_preserved": bool(chip_keeps_saved_free_transfers(rules.wildcard_ft_rule)),
    }


def price_engine_matches_rules(rules: SeasonRules) -> tuple[bool, str]:
    """Fail-fast drift check against our integer-tenths selling-price engine.

    The engine (repositories/metrics) implements, for purchase price P and
    official market C: sell = C when C <= P, else P + floor((C-P)*fee).  This
    matches the official rule set exactly when the sell-on fee is 0.5 and
    falling prices return the official market price (not the purchase price).
    """

    fee_is_half = abs(rules.sell_on_fee - 0.5) < 1e-9
    falling_market = rules.sell_at_purchase_price_when_falling is False
    if fee_is_half and falling_market:
        return True, "selling-price engine matches official settings"
    parts = []
    if not fee_is_half:
        parts.append(f"sell_on_fee {rules.sell_on_fee} != 0.5")
    if not falling_market:
        parts.append("element_sell_at_purchase_price=true is not represented by the engine")
    return False, "; ".join(parts)
