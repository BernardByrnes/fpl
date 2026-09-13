"""Phase 8A — candidate universe (deterministic transfer-target generation).

Answers ONLY "which players are valid candidates for the optimizer to consider?".
It does not rank final routes, does not recommend or execute a transfer, runs no
Monte Carlo/policy search/route comparison, and never affects predictive models.

Scoring basis is CORE (Phase 5/6 exclude stochastic bonus); expected CORE is taken
from the accepted ``core_xpts`` component, never TOTAL_PROXY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import transfer_state as ts
from .season_rules import window_flag

PHASE8A_VERSION = "candidate_universe_v8a_1.0.0"

CANONICAL_UNSUPPORTED_FLAG = "CANONICAL_H4_H6_UNSUPPORTED"
SCORE_BASIS = "CORE"
VALUE_LABEL = "DESCRIPTIVE_VALUE_ONLY"

POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
POSITIONS = ("GKP", "DEF", "MID", "FWD")

PREDICTED = "PREDICTED"
NO_FIXTURE = "NO_FIXTURE"
MISSING_PROJECTION = "MISSING_PROJECTION"

#: Discovery-completeness diagnostics.
OFFICIAL_PLAYER_POOL_INCOMPLETE = "OFFICIAL_PLAYER_POOL_INCOMPLETE"
CANDIDATE_PREDICTIVE_SUPPORT_MISSING = "CANDIDATE_PREDICTIVE_SUPPORT_MISSING"


class DiscoveryCompletenessError(ValueError):
    """An official-pool player is neither an eligible candidate nor excluded with a reason."""


class PredictiveSupportMissing(ValueError):
    """A player with a fixture in the window has no certified predictive support."""


CRITERIA = ("TOP_FIRST_EVENT_CORE", "TOP_WINDOW_CORE", "TOP_MINUTES", "TOP_CHEAP", "TOP_VALUE")


@dataclass(frozen=True)
class CandidateConfig:
    top_n_per_criterion: int = 20
    value_label: str = VALUE_LABEL

    def as_dict(self) -> dict[str, Any]:
        return {"top_n_per_criterion": int(self.top_n_per_criterion), "value_label": self.value_label}


@dataclass(frozen=True)
class EventFeature:
    event: int
    status: str
    expected_core: float
    expected_minutes: float
    p_start: float
    p_60_plus: float
    availability: float
    fixture_count: int
    partial: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event), "status": self.status,
            "expected_core": round(self.expected_core, 6),
            "expected_minutes": round(self.expected_minutes, 6),
            "p_start": round(self.p_start, 6), "p_60_plus": round(self.p_60_plus, 6),
            "availability": round(self.availability, 6), "fixture_count": int(self.fixture_count),
            "partial": bool(self.partial),
        }


# ---------------------------------------------------------------------------
# Loaders (read-only)
# ---------------------------------------------------------------------------


def load_pool(conn) -> dict[int, dict[str, Any]]:
    from .scoring_rules import POSITION_IDS

    pool: dict[int, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT id, element_type, team_id, web_name, full_name FROM players WHERE is_active=1"
    ):
        position = POSITION_IDS.get(int(row["element_type"])) if row["element_type"] is not None else None
        pool[int(row["id"])] = {
            "player_id": int(row["id"]), "position": position,
            "club_id": None if row["team_id"] is None else int(row["team_id"]),
            "web_name": row["web_name"], "full_name": row["full_name"],
        }
    return pool


def load_fixtures_by_team(conn, events: Sequence[int]) -> dict[tuple[int, int], list[int]]:
    fixtures: dict[tuple[int, int], list[int]] = {}
    for row in conn.execute(
        "SELECT id, event, team_h, team_a FROM fixtures WHERE event IN (%s)" % ",".join("?" for _ in events),
        [int(e) for e in events],
    ):
        event = int(row["event"])
        fixtures.setdefault((event, int(row["team_h"])), []).append(int(row["id"]))
        fixtures.setdefault((event, int(row["team_a"])), []).append(int(row["id"]))
    for key in fixtures:
        fixtures[key].sort()
    return fixtures


def load_projection_rows(conn, run_id: int, kind: str | None = None) -> dict[tuple[int, int], dict[str, Any]]:
    from . import analytics

    rows: dict[tuple[int, int], dict[str, Any]] = {}
    if kind is None:
        for record in analytics.xpts_projections(conn, int(run_id)):
            rows[(int(record["player_id"]), int(record["fixture_id"]))] = record["payload"]
    else:
        for record in analytics.frozen_predictions(conn, int(run_id), [kind]):
            rows[(int(record["player_id"]), int(record["fixture_id"]))] = record["payload"]
    return rows


def latest_price_snapshot(conn, event: int) -> ts.PriceSnapshot:
    """UNBOUNDED price snapshot.  Not causal -- do not use for a decision.

    Kept only so existing diagnostics keep working.  It scans every
    player_snapshots row with no cutoff and no ordering, so the "latest" price it
    returns is an artefact of scan order.  Transfer decisions must use
    :func:`price_snapshot_as_of`.
    """

    prices = {
        int(row["player_id"]): int(row["now_cost"])
        for row in conn.execute("SELECT player_id, now_cost FROM player_snapshots")
        if row["now_cost"] is not None
    }
    return ts.PriceSnapshot(event=int(event), prices=prices)


def price_snapshot_as_of(
    conn,
    event: int,
    cutoff: str,
    *,
    required_player_ids: Iterable[int] | None = None,
) -> ts.PriceSnapshot:
    """Causal price snapshot: each player's price strictly at or before ``cutoff``.

    Delegates to :func:`fpl_brain.causality.price_snapshot_as_of`, which picks the
    latest eligible row per player (``captured_at DESC, id DESC``) and never lets a
    later price leak backward.  Fails closed when a required player has no causal
    price, because a decision must not be priced from the future.
    """

    from .causality import (
        DIAG_PRICE_MISSING_AS_OF_CUTOFF,
        CausalityError,
        price_snapshot_as_of as _resolve,
    )

    resolved = _resolve(conn, cutoff, event=int(event), fallback_to_earliest=False)
    if required_player_ids is not None:
        missing = [int(pid) for pid in required_player_ids if int(pid) not in resolved.prices]
        if missing:
            raise CausalityError(
                f"{DIAG_PRICE_MISSING_AS_OF_CUTOFF}: no official price at or before {cutoff} for "
                f"{missing}"
            )
    prices = dict(resolved.prices)
    return ts.PriceSnapshot(
        event=int(event), prices=prices,
        snapshot_id=ts.price_snapshot_identity(int(event), prices),
    )


# ---------------------------------------------------------------------------
# Pure builder
# ---------------------------------------------------------------------------


def _event_feature(
    player_id: int,
    club_id: int,
    event: int,
    events_fixtures: Mapping[tuple[int, int], Sequence[int]],
    xpts_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    minutes_rows: Mapping[tuple[int, int], Mapping[str, Any]],
) -> EventFeature:
    fixture_ids = [int(fid) for fid in events_fixtures.get((int(event), int(club_id)), [])]
    if not fixture_ids:
        return EventFeature(event, NO_FIXTURE, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    x_rows = [xpts_rows[(player_id, fid)] for fid in fixture_ids if (player_id, fid) in xpts_rows]
    m_rows = [minutes_rows[(player_id, fid)] for fid in fixture_ids if (player_id, fid) in minutes_rows]
    if not x_rows:
        return EventFeature(event, MISSING_PROJECTION, 0.0, 0.0, 0.0, 0.0, 0.0, len(fixture_ids))

    expected_core = sum(float(row.get("core_xpts") or 0.0) for row in x_rows)
    expected_minutes = sum(float(row.get("expected_minutes") or 0.0) for row in x_rows)
    p_start = max(float(row.get("p_start") or 0.0) for row in x_rows)
    p_60_plus = max(float(row.get("p_60_plus") or 0.0) for row in x_rows)
    availability_values = [
        float(row.get("joint_availability") or row.get("p_available") or 0.0) for row in m_rows
    ]
    availability = max(availability_values) if availability_values else max(
        float(row.get("p_appearance") or 0.0) for row in x_rows
    )
    partial = len(x_rows) < len(fixture_ids)
    return EventFeature(event, PREDICTED, expected_core, expected_minutes, p_start, p_60_plus,
                        availability, len(fixture_ids), partial=partial)


def build_universe(
    *,
    pool: Mapping[int, dict[str, Any]],
    events_fixtures: Mapping[tuple[int, int], Sequence[int]],
    xpts_rows_by_event: Mapping[int, Mapping[tuple[int, int], Mapping[str, Any]]],
    minutes_rows_by_event: Mapping[int, Mapping[tuple[int, int], Mapping[str, Any]]],
    events: Sequence[int],
    owned_ids: Iterable[int],
    price_snapshot: ts.PriceSnapshot,
    config: CandidateConfig | None = None,
    planning_cutoff: str | None = None,
    run_refs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full candidate universe plus the transparent search view."""

    config = config or CandidateConfig()
    events = [int(e) for e in events]
    owned = {int(pid) for pid in owned_ids}
    # One payload per (event, player) for risk-flag metadata only.
    risk_index: dict[int, dict[int, Mapping[str, Any]]] = {}
    for event in events:
        index: dict[int, Mapping[str, Any]] = {}
        for (pid, _fid), payload in xpts_rows_by_event.get(event, {}).items():
            index.setdefault(int(pid), payload)
        risk_index[event] = index
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for player_id in sorted(pool):
        meta = pool[player_id]
        position = meta.get("position")
        club_id = meta.get("club_id")
        price = price_snapshot.price(int(player_id))
        if position not in POSITION_ORDER:
            excluded.append({"player_id": int(player_id), "reason": "INVALID_POSITION"})
            continue
        if club_id is None:
            excluded.append({"player_id": int(player_id), "reason": "MISSING_CLUB"})
            continue
        if price is None:
            excluded.append({"player_id": int(player_id), "reason": "MISSING_CURRENT_PRICE"})
            continue
        features = [
            _event_feature(int(player_id), int(club_id), event, events_fixtures,
                           xpts_rows_by_event.get(event, {}), minutes_rows_by_event.get(event, {}))
            for event in events
        ]
        risk_flags: list[str] = []
        for event in events:
            payload = risk_index[event].get(int(player_id))
            if isinstance(payload, Mapping):
                risk_flags.extend(str(flag) for flag in (payload.get("risk_flags") or []))
        total_core = sum(feature.expected_core for feature in features)
        total_minutes = sum(feature.expected_minutes for feature in features)
        rows.append({
            "player_id": int(player_id),
            "position": position,
            "club_id": int(club_id),
            "web_name": meta.get("web_name"),
            "full_name": meta.get("full_name"),
            "owned": bool(int(player_id) in owned),
            "current_market_price_tenths": int(price),
            "events": features,
            "supported_3gw_expected_core": total_core,
            "supported_3gw_expected_minutes": total_minutes,
            "mean_expected_core_per_event": (total_core / len(events)) if events else 0.0,
            "descriptive_value": (total_core / (price / 10.0)) if price else None,
            "value_label": VALUE_LABEL,
            "risk_flags": sorted(set(risk_flags)),
            "inclusion_reasons": [],
            "search_view_dominated_by": None,
        })

    rows.sort(key=lambda row: (POSITION_ORDER[row["position"]], int(row["player_id"])))
    _fill_search_view(rows, config)
    _mark_safe_dominance(rows)

    audit = _completeness_audit(rows, events)
    search_view = [
        {"player_id": row["player_id"], "position": row["position"],
         "inclusion_reasons": list(row["inclusion_reasons"]),
         "search_view_dominated_by": row["search_view_dominated_by"]}
        for row in rows if row["inclusion_reasons"]
    ]
    counts_by_position = {position: sum(1 for row in rows if row["position"] == position) for position in POSITIONS}
    search_by_position = {position: sum(1 for row in search_view if row["position"] == position) for position in POSITIONS}
    return {
        "phase_version": PHASE8A_VERSION,
        "planning_cutoff": planning_cutoff,
        "supported_events": events,
        "decision_horizon_length": len(events),
        "predictive_runs": dict(run_refs or {}),
        "price_snapshot_id": price_snapshot.identity(),
        "score_basis": SCORE_BASIS,
        # The window flag is DERIVED from the real decision events; a GW5-GW8
        # universe is no longer stamped as a GW4-GW6 window.
        "flags": [window_flag(events), CANONICAL_UNSUPPORTED_FLAG, VALUE_LABEL],
        "config": config.as_dict(),
        "universe": rows,
        "universe_count": len(rows),
        "counts_by_position": counts_by_position,
        "search_view": search_view,
        "search_view_count": len(search_view),
        "search_view_counts_by_position": search_by_position,
        "excluded": excluded,
        "completeness_audit": audit,
        "no_recommendation": True,
    }


def _search_view_rankings(group: list[dict[str, Any]], n: int) -> dict[str, list[dict[str, Any]]]:
    """The five Phase-8A search-view criteria, ranked, for one position group.

    Single source for both the reason-labelled view (``_fill_search_view``) and
    the id-set view (``search_view_ids``) so the two can never disagree.  The
    first criterion is POSITIONAL (``_event_core(row, 0)``): ``row["events"]`` is
    built in decision-event order, so index 0 is that window's first event for
    any window.  Looking the event up by NUMBER would be silently zero off GW4.
    """

    return {
        "TOP_FIRST_EVENT_CORE": sorted(group, key=lambda row: (
            -_event_core(row, 0), row["current_market_price_tenths"], row["player_id"]))[:n],
        "TOP_WINDOW_CORE": sorted(group, key=lambda row: (
            -row["supported_3gw_expected_core"], row["current_market_price_tenths"], row["player_id"]))[:n],
        "TOP_MINUTES": sorted(group, key=lambda row: (
            -row["supported_3gw_expected_minutes"], row["current_market_price_tenths"], row["player_id"]))[:n],
        "TOP_CHEAP": sorted(group, key=lambda row: (row["current_market_price_tenths"], row["player_id"]))[:n],
        "TOP_VALUE": sorted(group, key=lambda row: (
            -(row["descriptive_value"] if row["descriptive_value"] is not None else float("-inf")),
            row["player_id"]))[:n],
    }


def search_view_ids(rows: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]], n: int) -> set[int]:
    """The Phase-8A search view as a set of player ids at any N.

    Accepts either the universe's row list or a ``player_id -> row`` mapping, so
    it can be compared directly with ``route_optimizer.search_view_ids``.  The
    two are asserted equal in the tests for every window.
    """

    row_list = list(rows.values()) if isinstance(rows, Mapping) else list(rows)
    view: set[int] = set()
    for position in POSITIONS:
        group = [row for row in row_list if row["position"] == position]
        if not group:
            continue
        for ranked in _search_view_rankings(group, int(n)).values():
            for row in ranked:
                view.add(int(row["player_id"]))
    return view


def _fill_search_view(rows: list[dict[str, Any]], config: CandidateConfig) -> None:
    n = int(config.top_n_per_criterion)
    for position in POSITIONS:
        group = [row for row in rows if row["position"] == position]
        if not group:
            continue
        for criterion, ranked in _search_view_rankings(group, n).items():
            for row in ranked:
                reason = f"{criterion}_{position}"
                if reason not in row["inclusion_reasons"]:
                    row["inclusion_reasons"].append(reason)
    for row in rows:
        row["inclusion_reasons"].sort()


def _event_core(row: Mapping[str, Any], index: int) -> float:
    features = row["events"]
    return float(features[index].expected_core) if index < len(features) else 0.0


def _mark_safe_dominance(rows: list[dict[str, Any]]) -> None:
    """Strict, guaranteed-safe marking only (same position AND same club)."""

    by_group: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault((row["position"], row["club_id"]), []).append(row)
    for group in by_group.values():
        for candidate in group:
            dominators = [
                other["player_id"] for other in group
                if other["player_id"] != candidate["player_id"]
                and other["current_market_price_tenths"] <= candidate["current_market_price_tenths"]
                and all(
                    _event_core(other, i) >= _event_core(candidate, i)
                    for i in range(len(candidate["events"]))
                )
                and all(
                    float(other["events"][i].expected_minutes) >= float(candidate["events"][i].expected_minutes)
                    for i in range(len(candidate["events"]))
                )
                and (
                    other["current_market_price_tenths"] < candidate["current_market_price_tenths"]
                    or any(_event_core(other, i) > _event_core(candidate, i) for i in range(len(candidate["events"])))
                    or any(
                        float(other["events"][i].expected_minutes) > float(candidate["events"][i].expected_minutes)
                        for i in range(len(candidate["events"]))
                    )
                )
            ]
            if dominators:
                candidate["search_view_dominated_by"] = int(min(dominators))


def _completeness_audit(rows: Sequence[Mapping[str, Any]], events: Sequence[int]) -> dict[str, Any]:
    audit: dict[str, Any] = {"by_event": {}, "missing_projection_players": []}
    for index, event in enumerate(events):
        statuses = {"PREDICTED": 0, "PREDICTED_PARTIAL": 0, NO_FIXTURE: 0, MISSING_PROJECTION: 0}
        for row in rows:
            feature = row["events"][index]
            if feature.status == PREDICTED and feature.partial:
                statuses["PREDICTED_PARTIAL"] += 1
            else:
                statuses[feature.status] = statuses.get(feature.status, 0) + 1
            if feature.status == MISSING_PROJECTION:
                audit["missing_projection_players"].append({"player_id": row["player_id"], "event": int(event)})
        audit["by_event"][str(int(event))] = statuses
    audit["missing_projection_players"] = sorted(
        audit["missing_projection_players"], key=lambda item: (item["event"], item["player_id"])
    )
    return audit


def unresolved_predictive_support_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Players that HAVE a fixture in the window but no certified projection.

    This is the ``MISSING_PREDICTION`` case and it must never be confused with an
    EXPLICIT ZERO:

    * EXPLICIT ZERO — ``NO_FIXTURE``: the player legitimately has no fixture
      (blank Gameweek), so a zero is the correct, valid value.
    * MISSING PREDICTION — the player has a fixture but no certified predictive
      row.  The stored ``expected_core`` is 0.0 only because nothing was
      supplied, NOT because the model predicted zero.  Treating it as a valid
      zero would let a newly added or unsupported player look like a
      legitimate zero-value candidate.

    Returns one record per (player, event) that is missing support.
    """

    unresolved: list[dict[str, Any]] = []
    for row in rows:
        for feature in row["events"]:
            if feature.status == MISSING_PROJECTION and int(feature.fixture_count) > 0:
                unresolved.append({
                    "player_id": int(row["player_id"]),
                    "event": int(feature.event),
                    "fixture_count": int(feature.fixture_count),
                    "reason": CANDIDATE_PREDICTIVE_SUPPORT_MISSING,
                })
    return sorted(unresolved, key=lambda item: (item["event"], item["player_id"]))


def _ids_digest(player_ids: Iterable[int]) -> str:
    import hashlib

    ordered = sorted({int(pid) for pid in player_ids})
    return hashlib.sha256(",".join(str(pid) for pid in ordered).encode()).hexdigest()[:32]


def discovery_completeness(
    *,
    pool: Mapping[int, dict[str, Any]],
    universe_rows: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    replacement_edges: Sequence[Mapping[str, Any]],
    screen: Mapping[str, Any] | None = None,
    persisted_pool_ids: Iterable[int] | None = None,
    official_pool_ids: Iterable[int] | None = None,
    enforce: bool = True,
) -> dict[str, Any]:
    """Auditable all-player discovery accounting, with hard assertions.

    Contract (permanent): every eligible official player receives a numerical
    opportunity BEFORE route pruning.  This function proves it by accounting for
    every official player exactly once — either as an eligible candidate or as an
    explicit, single-reason exclusion — and by checking that every legal
    replacement edge was numerically screened.

    Raises ``PredictiveSupportMissing`` then ``DiscoveryCompletenessError`` when
    the contract is broken, so a decision cannot proceed on an incomplete
    discovery set.
    """

    official_ids = {int(pid) for pid in (official_pool_ids if official_pool_ids is not None else pool)}
    persisted_ids = {int(pid) for pid in (persisted_pool_ids if persisted_pool_ids is not None else pool)}
    eligible_ids = {int(row["player_id"]) for row in universe_rows}
    priced_ids = {int(row["player_id"]) for row in universe_rows
                  if row.get("current_market_price_tenths") is not None}
    projected_ids = {
        int(row["player_id"]) for row in universe_rows
        if any(feature.status in (PREDICTED, NO_FIXTURE) for feature in row["events"])
    }

    by_reason: dict[str, int] = {}
    exclusion_by_player: dict[int, list[str]] = {}
    for entry in excluded:
        reason = str(entry.get("reason"))
        by_reason[reason] = by_reason.get(reason, 0) + 1
        exclusion_by_player.setdefault(int(entry["player_id"]), []).append(reason)

    enumerated_edges = len(list(replacement_edges))
    legal_edges = sum(1 for edge in replacement_edges if edge.get("currently_legal_single_transfer"))
    screened_edges = int((screen or {}).get("screened_legal_actions", legal_edges))
    promoted_edges_list = list((screen or {}).get("promotion_pool") or [])
    promoted_edges = int((screen or {}).get("promotion_pool_size", len(promoted_edges_list)))
    promoted_ids = {int(edge["in_player_id"]) for edge in promoted_edges_list}
    # Only a SCREENED edge carries the screen metric, so its presence proves the
    # promoted edge came out of numerical screening rather than around it.
    promoted_from_screening = all(
        edge.get("four_gw_proxy_delta") is not None for edge in promoted_edges_list
    )

    accounted = eligible_ids | set(exclusion_by_player)
    unaccounted = sorted(official_ids - accounted)
    multi_reason = sorted(pid for pid, reasons in exclusion_by_player.items() if len(reasons) > 1)
    eligible_and_excluded = sorted(eligible_ids & set(exclusion_by_player))

    unresolved = unresolved_predictive_support_rows(universe_rows)

    report: dict[str, Any] = {
        "official_pool_players": len(official_ids),
        "persisted_pool_players": len(persisted_ids),
        "priced_players": len(priced_ids),
        "projected_players": len(projected_ids),
        "eligible_candidate_players": len(eligible_ids),
        "excluded_players_by_reason": by_reason,
        "enumerated_edges": enumerated_edges,
        "legal_edges": legal_edges,
        "screened_edges": screened_edges,
        "promoted_edges": promoted_edges,
        "official_pool_ids_sha256": _ids_digest(official_ids),
        "eligible_candidate_ids_sha256": _ids_digest(eligible_ids),
        "promoted_in_ids_sha256": _ids_digest(promoted_ids),
        "unaccounted_official_players": unaccounted,
        "excluded_with_multiple_reasons": multi_reason,
        "eligible_and_excluded": eligible_and_excluded,
        "unresolved_predictive_support": unresolved,
        "assertions": {
            "A_every_official_player_eligible_or_excluded_once": not unaccounted and not multi_reason
            and not eligible_and_excluded,
            "B_every_legal_edge_screened": bool((screen or {}).get(
                "coverage", {}).get("every_legal_edge_screened", screened_edges == legal_edges)),
            "C_screened_edges_equals_legal_edges": screened_edges == legal_edges,
            "D_promotion_pool_subset_of_screened": promoted_from_screening
            and len(promoted_edges_list) <= screened_edges,
            "E_no_unresolved_predictive_support": not unresolved,
        },
        "no_recommendation": True,
    }

    if enforce:
        if unresolved:
            first = unresolved[:5]
            raise PredictiveSupportMissing(
                f"{CANDIDATE_PREDICTIVE_SUPPORT_MISSING}: {len(unresolved)} player/event pair(s) have a "
                f"fixture in the decision window but no certified predictive support, e.g. {first}. "
                "A missing projection is not a valid zero."
            )
        failed = [name for name, ok in report["assertions"].items() if not ok]
        if failed:
            raise DiscoveryCompletenessError(
                f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: discovery completeness assertion(s) failed: {failed}. "
                f"unaccounted={unaccounted[:8]} multi_reason={multi_reason[:8]} "
                f"eligible_and_excluded={eligible_and_excluded[:8]} "
                f"edges enumerated={enumerated_edges} legal={legal_edges} screened={screened_edges} "
                f"official={len(official_ids)} eligible={len(eligible_ids)}"
            )
    return report


# ---------------------------------------------------------------------------
# Replacement edges (Phase-7A legality only)
# ---------------------------------------------------------------------------


def build_replacement_edges(
    *,
    universe_rows: Sequence[Mapping[str, Any]],
    owned_ids: Iterable[int],
    state: ts.RouteState,
    price_snapshot: ts.PriceSnapshot,
    player_meta: Mapping[int, ts.PlayerMeta],
) -> list[dict[str, Any]]:
    """owned -> same-position, not-owned candidate; legality from Phase-7A only."""

    owned = {int(pid) for pid in owned_ids}
    owned_players = {int(row["player_id"]): row for row in universe_rows if int(row["player_id"]) in owned}
    edges: list[dict[str, Any]] = []
    for out_id in sorted(owned_players):
        out_row = owned_players[out_id]
        for candidate in universe_rows:
            in_id = int(candidate["player_id"])
            if in_id in owned or candidate["position"] != out_row["position"]:
                continue
            batch = ts.TransferBatch((ts.TransferAction(out_id, in_id),))
            result = ts.apply_transfer_batch(state, batch, price_snapshot, player_meta)
            edges.append({
                "out_player_id": out_id,
                "in_player_id": in_id,
                "position": out_row["position"],
                "currently_legal_single_transfer": bool(result.ok),
                "failure_reason": None if result.ok else "; ".join(result.errors),
                "bank_after_tenths": int(result.bank_after_tenths),
                "hit_points": int(result.hit_points),
                "incoming_price_tenths": int(candidate["current_market_price_tenths"]),
            })
    edges.sort(key=lambda edge: (edge["out_player_id"], edge["in_player_id"]))
    return edges


def jsonable(value: Any) -> Any:
    """Convert the universe (with EventFeature dataclasses) to JSON-safe data."""

    if isinstance(value, EventFeature):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def universe_metadata(universe: Mapping[str, Any]) -> dict[str, Any]:
    """Minimal DecisionPacket section (metadata + search-view summary only)."""

    return {
        "phase_version": universe.get("phase_version"),
        "planning_cutoff": universe.get("planning_cutoff"),
        "supported_events": universe.get("supported_events"),
        "score_basis": universe.get("score_basis"),
        "universe_count": universe.get("universe_count"),
        "counts_by_position": universe.get("counts_by_position"),
        "search_view_count": universe.get("search_view_count"),
        "search_view_counts_by_position": universe.get("search_view_counts_by_position"),
        "price_snapshot_id": universe.get("price_snapshot_id"),
        "risk_flags": universe.get("flags"),
        "completeness_audit": universe.get("completeness_audit"),
        "excluded_players_by_reason": _excluded_by_reason(universe.get("excluded") or []),
        "excluded_count": len(universe.get("excluded") or []),
        "no_recommendation": True,
    }


def _excluded_by_reason(excluded: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in excluded:
        reason = str(entry.get("reason"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts
