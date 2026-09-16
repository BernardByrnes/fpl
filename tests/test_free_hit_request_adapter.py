"""Free Hit V1 — production manager-state authority.

The permanent world must be CANONICAL: a caller cannot assert a squad, a bank, a
basis, a position label, a club or a price.  These tests drive a real temporary
store so the authority is exercised end to end rather than asserted on a
hand-made object.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import chip_decision as cd  # noqa: E402
from fpl_brain import chip_free_hit as fh  # noqa: E402
from fpl_brain import free_hit_request_adapter as ad  # noqa: E402
from fpl_brain import repositories as repo  # noqa: E402
from fpl_brain import season_rules as sr  # noqa: E402
from fpl_brain.chip_wildcard import WildcardPoolBinding, WildcardPredictiveIdentity  # noqa: E402
from fpl_brain.database import connect_database  # noqa: E402
from fpl_brain.ingest_provenance import element_id_sha256  # noqa: E402
from fpl_brain.models import (  # noqa: E402
    EventRecord, PickRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord,
)

ENTRY = 241392
EVENT = 5
WORLDS = 6
CUTOFF = "2026-09-16T11:00:00Z"
CAPTURED_AT = "2026-09-16T08:00:00Z"
DATA = "sha256:" + "d" * 64
SOURCE = "sha256:" + "s" * 64
CERT = "sha256:" + "c" * 64
CONFIG = "sha256:" + "f" * 64
RULES = sr.SeasonRules(season="2026/27")

UNIVERSE = tuple(range(1, 31))
POSITION = {pid: ("GKP" if pid <= 4 else "DEF" if pid <= 14 else "MID" if pid <= 24 else "FWD")
            for pid in UNIVERSE}
POSITION_ID = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
CLUB = {pid: 100 + ((pid - 1) % 10) for pid in UNIVERSE}
OWNED = (1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27)
XI = (1, 5, 6, 7, 15, 16, 17, 18, 25, 26, 27)
BENCH_GK, BENCH_OUT = 2, (8, 9, 19)
CAPTAIN, VICE = 25, 26


def _seed(conn, *, prices: dict[int, int] | None = None, bank: int = 20, ft: int = 2,
          with_generation: bool = True) -> None:
    prices = prices if prices is not None else {pid: 50 for pid in UNIVERSE}
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=t, name=f"Team {t}") for t in sorted(set(CLUB.values()))])
        repo.upsert_positions(conn, [
            PositionRecord(id=1, singular_name="Goalkeeper", singular_name_short="GKP"),
            PositionRecord(id=2, singular_name="Defender", singular_name_short="DEF"),
            PositionRecord(id=3, singular_name="Midfielder", singular_name_short="MID"),
            PositionRecord(id=4, singular_name="Forward", singular_name_short="FWD"),
        ])
        repo.upsert_events(conn, [
            EventRecord(id=event, finished=1 if event < EVENT else 0,
                        data_checked=1 if event < EVENT else 0,
                        deadline_time=f"2026-09-{event + 11:02d}T12:30:00Z", raw_json={})
            for event in range(1, 39)
        ])
        repo.upsert_players(conn, [
            PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}", team_id=CLUB[pid],
                         element_type=POSITION_ID[POSITION[pid]])
            for pid in UNIVERSE
        ])
        run = repo.create_fetch_run(conn, "fetch_fpl", started_at=CAPTURED_AT)
        repo.insert_snapshots(conn, [
            PlayerSnapshotRecord(player_id=pid, captured_at=CAPTURED_AT,
                                 now_cost=int(prices[pid]), raw_json={})
            for pid in UNIVERSE if pid in prices
        ], run)
        repo.finish_fetch_run(conn, run, "success", current_event=EVENT)

        conn.execute("DELETE FROM chip_definitions")
        for index, name in enumerate(("freehit", "wildcard", "bboost", "3xc"), start=1):
            conn.execute(
                "INSERT INTO chip_definitions(id, name, number, chip_type, start_event, stop_event, "
                "updated_at) VALUES (?,?,?,?,?,?,?)",
                (index, name, 1, "team", 2, 38, CAPTURED_AT),
            )
        if with_generation:
            ids = sorted(UNIVERSE)
            conn.execute(
                """INSERT INTO bootstrap_generations(fetch_run_id, captured_at, accepted,
                   official_element_count, parsed_count, persisted_count, element_ids_sha256,
                   element_ids_json, acceptance_rule, acceptance_rule_version, recorded_at)
                   VALUES (?,?,1,?,?,?,?,?,?,?,?)""",
                (run, CAPTURED_AT, len(ids), len(ids), len(ids), element_id_sha256(ids),
                 json.dumps(ids), "test", "v1", CAPTURED_AT),
            )
        picks = (*XI, BENCH_GK, *BENCH_OUT)
        repo.upsert_squad_picks(conn, ENTRY, EVENT, [
            PickRecord(player_id=pid, position=slot, is_captain=1 if pid == CAPTAIN else 0,
                       is_vice_captain=1 if pid == VICE else 0,
                       multiplier=2 if pid == CAPTAIN else (1 if slot <= 11 else 0), raw_json={})
            for slot, pid in enumerate(picks, start=1)
        ])
        already = {int(row["player_id"]) for row in repo.active_manager_acquisitions(conn, ENTRY)}
        for pid in OWNED:
            if pid in already:
                continue
            repo.insert_manager_acquisition(conn, ENTRY, pid, 1, 50,
                                            source="official_transfer_history", created_at=CAPTURED_AT)
        conn.execute(
            """INSERT INTO manager_state(entry_id, fetch_run_id, captured_at, event, bank, team_value,
               total_transfers, event_transfers, event_transfers_cost, points_on_bench, active_chip,
               free_transfers_manual, raw_json)
               VALUES (?, ?, ?, ?, ?, 750, 0, 0, 0, 0, NULL, ?, ?)""",
            (ENTRY, run, CAPTURED_AT, EVENT, int(bank), int(ft),
             json.dumps({"transfers_endpoint_available": True, "transfers": [], "history": {"current": []}})),
        )
        # The event-start free-transfer bank is a CANONICAL EXPLICIT record: it is
        # only ever read from a manual observation, never inferred from a possibly
        # depleted current count.
        repo.upsert_manual_manager_state(
            conn, ENTRY, EVENT, int(ft), int(bank),
            source="user_confirmed_free_hit_test_state", captured_at=CAPTURED_AT,
            event_start_free_transfers=int(ft),
        )


@pytest.fixture
def conn(tmp_path):
    connection = connect_database(tmp_path / "fpl.db")
    try:
        yield connection
    finally:
        connection.close()


def _identity(**overrides) -> WildcardPredictiveIdentity:
    base = dict(cutoff=CUTOFF, data_snapshot_sha256=DATA, source_snapshot_sha256=SOURCE,
                generation=CAPTURED_AT, model_config_identity=CONFIG)
    base.update(overrides)
    return WildcardPredictiveIdentity(**base)


def _h1_worlds(values: dict[int, float] | None = None, *, identity=None, event: int = EVENT):
    from fpl_brain.chip_wildcard import WildcardWorldInputs

    values = values if values is not None else {pid: 1.0 for pid in UNIVERSE}
    core = {pid: tuple(float(values[pid]) for _ in range(WORLDS)) for pid in UNIVERSE}
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in UNIVERSE}
    return WildcardWorldInputs(
        event=int(event), worlds=WORLDS, player_ids=UNIVERSE, minutes=minutes, core=core,
        identity=identity if identity is not None else _identity(),
    )


def _authority(**overrides) -> fh.FreeHitDecisionAuthority:
    """The canonical certified decision context, from a self-consistent artifact."""

    from fpl_brain import four_gw_decision as fg

    base = dict(cutoff=CUTOFF, data=DATA, source=SOURCE, generation=CAPTURED_AT, config=CONFIG)
    base.update(overrides)
    artifact = {
        "planning_cutoff": base["cutoff"],
        "data_snapshot_sha256": base["data"],
        "certified_bundle_identity": {
            "source_snapshot_sha256": base["source"],
            "generation": base["generation"],
            "model_config_identity": base["config"],
        },
    }
    artifact["four_gw_certification_identity"] = fg.certification_identity_of(artifact)
    return fh.FreeHitDecisionAuthority.from_certification(artifact)


def _binding(*, event: int = EVENT, snapshot: str = DATA, identity=None):
    return cd.ChipHorizonBinding(
        planning_event=int(event), horizon_events=cd.canonical_chip_horizon(int(event)),
        certification_identity=(identity or _authority().certification_identity),
        data_snapshot_sha256=snapshot,
    )


def _pool() -> WildcardPoolBinding:
    return WildcardPoolBinding(
        generation_identity=CAPTURED_AT, generation_id_sha256=element_id_sha256(sorted(UNIVERSE)),
        official_count=len(UNIVERSE), eligible_ids=tuple(sorted(UNIVERSE)),
    )


def _tails(*, play_values=(10.0, 10.0, 10.0), save_values=(10.0, 10.0, 10.0)):
    from fpl_brain import chip_wildcard as _wc

    permanent = _permanent()
    events = (EVENT + 1, EVENT + 2, EVENT + 3)
    restored_ft = fh.post_free_hit_ft_state(RULES, event_start_free_transfers=2)
    save_ft = int(sr.free_transfers_after_gameweek(RULES, 2, 0))
    basis = permanent.purchase_price_tenths

    def _build(arm, values, free_transfers):
        return fh.FreeHitTailRoute(
            arm=arm,
            events=tuple(
                _wc.WildcardSaveRouteEvent(
                    event=int(event), squad_ids=tuple(sorted(OWNED)), bank_tenths=20,
                    purchase_price_tenths={int(k): int(v) for k, v in basis.items()},
                    free_transfers=int(free_transfers), mean_net_core=float(value),
                )
                for event, value in zip(events, values)
            ),
            h2_squad_ids=tuple(sorted(OWNED)), h2_bank_tenths=20,
            h2_purchase_price_tenths={int(k): int(v) for k, v in basis.items()},
            h2_free_transfers=int(free_transfers),
        )

    return (
        _build(fh.FreeHitTailRoute.FREE_HIT_ARM_PLAY, play_values, restored_ft),
        _build(fh.FreeHitTailRoute.FREE_HIT_ARM_SAVE, save_values, save_ft),
    )


def _permanent() -> fh.FreeHitPermanentState:
    return fh.FreeHitPermanentState(
        event=EVENT, owned_ids=tuple(sorted(OWNED)),
        purchase_price_tenths={pid: 50 for pid in OWNED}, bank_tenths=20,
        event_start_free_transfers=2, positions=POSITION, clubs=CLUB,
    )


def _certified(**overrides) -> ad.FreeHitCertifiedInputs:
    play_tail, save_tail = _tails()
    base = dict(horizon_binding=_binding(), h1_worlds=_h1_worlds(), world_identity=_identity(),
                pool_binding=_pool(), play_tail=play_tail, save_tail=save_tail,
                decision_authority=_authority())
    base.update(overrides)
    return ad.FreeHitCertifiedInputs(**base)


def _state(conn) -> ad.FreeHitManagerState:
    return ad.free_hit_manager_state(conn, ENTRY, EVENT, decision_cutoff=CUTOFF)


# ---------------------------------------------------------------------------
# CANONICAL AUTHORITY
# ---------------------------------------------------------------------------


def test_the_canonical_manager_state_is_sourced_and_accepted(conn):
    _seed(conn)
    state = _state(conn)
    assert state.owned_ids == tuple(sorted(OWNED))
    assert state.bank_tenths == 20
    assert state.event_start_free_transfers == 2
    assert state.problems() == []
    # Positions, clubs and PIT prices all come from canonical authority.
    assert state.positions[6] == "DEF" and state.positions[16] == "MID"
    assert state.clubs[1] == 100
    assert state.market_price_tenths[1] == 50
    assert state.market_price_basis.startswith("analytics.snapshot_as_of@")

    request = ad.build_free_hit_request(state, _certified(), conn=conn)
    evaluation = fh.evaluate_free_hit(request)
    assert evaluation.action == cd.CHIP_ACTION_FH
    assert evaluation.candidate_metrics["mean_paired_uplift"] is not None
    assert evaluation.candidate_metrics["restoration_leaks"] == []


def test_the_pool_binding_comes_from_the_accepted_generation_store(conn):
    _seed(conn)
    binding = ad.resolve_pool_binding(conn)
    assert binding.eligible_ids == tuple(sorted(UNIVERSE))
    assert binding.problems() == []


def test_a_rejected_or_absent_generation_refuses(conn):
    _seed(conn, with_generation=False)
    with pytest.raises(Exception) as caught:
        ad.resolve_pool_binding(conn)
    assert "no accepted official generation is recorded" in str(caught.value)


def test_a_missing_canonical_price_refuses(conn):
    """An official player with no point-in-time price is a contradiction, not a drop.

    The exhaustive-discovery contract does not permit silently discarding an
    eligible player because his price is missing, so the universe is incomplete
    and the whole request refuses.
    """

    _seed(conn, prices={pid: 50 for pid in UNIVERSE if pid != 16})
    state = _state(conn)
    assert 16 not in state.market_price_tenths
    problems = state.problems()
    assert any("market prices omit 1 official player" in problem for problem in problems), problems
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(state, _certified(), conn=conn)
    assert caught.value.reasons[0] == ad.FH_MANAGER_STATE_MISSING
    assert "market prices omit" in str(caught.value)


def test_a_post_cutoff_price_is_invisible_to_a_historical_replay(conn):
    """Prices are as of the cutoff: a later snapshot cannot leak into the decision."""

    _seed(conn)
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl", started_at="2026-09-17T08:00:00Z")
        repo.insert_snapshots(conn, [
            PlayerSnapshotRecord(player_id=1, captured_at="2026-09-17T08:00:00Z", now_cost=999, raw_json={})
        ], run)
        repo.finish_fetch_run(conn, run, "success", current_event=EVENT)
    state = _state(conn)
    assert state.market_price_tenths[1] == 50, "a post-cutoff price leaked into the decision"


def test_building_without_canonical_authority_refuses_by_default(conn):
    _seed(conn)
    state = _state(conn)
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(state, _certified())
    assert caught.value.reasons[0] == ad.FH_CANONICAL_AUTHORITY_REQUIRED


# ---------------------------------------------------------------------------
# CALLER FABRICATION ATTACKS
# ---------------------------------------------------------------------------


def _forged(conn, **overrides) -> ad.FreeHitManagerState:
    import dataclasses
    return dataclasses.replace(_state(conn), **overrides)


#: Each forgery is INTERNALLY CONSISTENT — it satisfies every local rule — so
#: only the canonical re-derivation can reveal it.
_FORGED_SQUAD = tuple(sorted((*OWNED[:-1], 28)))
_FORGED_BASIS = {**{pid: 50 for pid in _FORGED_SQUAD}}
_FORGED_BASIS[28] = 50


@pytest.mark.parametrize("overrides,needle", [
    ({"owned_ids": _FORGED_SQUAD, "purchase_price_tenths": _FORGED_BASIS}, "squad"),
    ({"bank_tenths": 999}, "bank"),
    ({"event_start_free_transfers": 5}, "free transfers"),
    ({"purchase_price_tenths": {pid: 1 for pid in OWNED}}, "basis"),
    ({"positions": {**POSITION, 6: "MID", 16: "DEF"}}, "position"),
    ({"clubs": {**CLUB, 6: 111}}, "club"),
    ({"market_price_tenths": {pid: 1 for pid in UNIVERSE}}, "price"),
])
def test_a_forged_permanent_component_refuses(conn, overrides, needle):
    """Every permanent fact is re-derived and compared, not trusted."""

    _seed(conn)
    forged = _forged(conn, **overrides)
    assert forged.problems() == [], "the forgery must pass every LOCAL rule"
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(forged, _certified(), conn=conn)
    assert caught.value.reasons[0] == ad.FH_CALLER_STATE_DISAGREES
    assert needle in str(caught.value), str(caught.value)


def test_a_forged_position_map_cannot_change_the_budget(conn):
    """Relabelling positions would change which temporaries are legal, and is refused."""

    _seed(conn)
    forged = _forged(conn, positions={**POSITION, 6: "MID", 16: "DEF"})
    assert forged.problems() == []          # locally it still looks like 2/5/5/3
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(forged, _certified(), conn=conn)
    assert caught.value.reasons[0] == ad.FH_CALLER_STATE_DISAGREES


# ---------------------------------------------------------------------------
# CERTIFIED INPUTS
# ---------------------------------------------------------------------------


def test_a_certified_input_that_is_not_the_bound_world_refuses(conn):
    _seed(conn)
    state = _state(conn)
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(
            state,
            _certified(h1_worlds=_h1_worlds(identity=_identity(cutoff="2026-09-16T12:00:00Z"))),
            conn=conn,
        )
    # The world no longer matches the CERTIFIED authority, not merely the
    # request's own companion object.
    assert caught.value.reasons[0] == fh.FH_DECISION_AUTHORITY_MISMATCH
    assert "cutoff" in str(caught.value)


def test_an_empty_snapshot_refuses_before_any_numeric_work(conn):
    """Every empty-snapshot route refuses, and the cause is named."""

    _seed(conn)
    state = _state(conn)
    cases = (
        ("binding snapshot empty", _certified(horizon_binding=_binding(snapshot=""))),
        ("world snapshot empty",
         _certified(h1_worlds=_h1_worlds(identity=_identity(data_snapshot_sha256="")))),
    )
    # An authority that carries no data snapshot cannot even be built: the anchor
    # is refused rather than becoming a hole in the anchoring.
    with pytest.raises(fh.FreeHitInputError) as anchor:
        _authority(data="")
    assert anchor.value.reasons[0] == fh.FH_DECISION_AUTHORITY_REQUIRED
    for label, certified in cases:
        with pytest.raises(ad.FreeHitAdapterError) as caught:
            ad.build_free_hit_request(state, certified, conn=conn)
        reasons = set(caught.value.reasons)
        assert reasons & {fh.FH_DATA_SNAPSHOT_REQUIRED, fh.FH_DECISION_AUTHORITY_REQUIRED,
                          fh.FH_DECISION_AUTHORITY_MISMATCH}, (label, reasons)
        assert "snapshot" in str(caught.value).lower(), (label, str(caught.value))


def test_a_horizon_bound_to_another_event_refuses(conn):
    _seed(conn)
    state = _state(conn)
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(
            state, _certified(horizon_binding=_binding(event=EVENT + 1)),
            conn=conn,
        )
    assert caught.value.reasons[0] == fh.FH_HORIZON_NOT_CANONICAL


def test_the_certified_path_reaches_the_evaluator_end_to_end(conn):
    """The production path produces a numeric review-only evaluation."""

    _seed(conn)
    state = _state(conn)
    values = {pid: 1.0 for pid in UNIVERSE}
    for pid in (20, 21, 22, 23, 24, 28, 29, 30):
        values[pid] = 6.0
    request = ad.build_free_hit_request(
        state, _certified(h1_worlds=_h1_worlds(values)), conn=conn,
    )
    evaluation = fh.evaluate_free_hit(request)
    assert evaluation.mean_uplift is not None and evaluation.mean_uplift > 0
    assert evaluation.execution_permitted is False
    assert evaluation.candidate_metrics["permanent_squad_unchanged_by_chip"] is True
    assert evaluation.candidate_metrics["normal_transfer_hits_charged"] == 0
    decision = cd.decide_chip_action(
        horizon_binding=request.horizon_binding,
        chip_availability=[{"name": "freehit", "available_for_event": True, "used": False,
                            "expired": False, "window": "GW2-GW38", "window_start_event": 2,
                            "window_stop_event": 38}],
        evaluations={cd.CHIP_ACTION_FH: evaluation}, certification_valid=True,
        manager_state={"squad_ids": list(state.owned_ids)},
    )
    assert decision.recommended_action == cd.CHIP_ACTION_FH
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
