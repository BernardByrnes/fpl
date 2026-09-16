"""Free Hit V1 — production manager-state authority.

The permanent world must be CANONICAL: a caller cannot assert a squad, a bank, a
basis, a position label, a club or a price.  These tests drive a real temporary
store so the authority is exercised end to end rather than asserted on a
hand-made object.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import tempfile
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpl_brain import chip_decision as cd  # noqa: E402
from fpl_brain import chip_free_hit as fh  # noqa: E402
from fpl_brain import free_hit_request_adapter as ad  # noqa: E402
from fpl_brain import four_gw_decision as fg  # noqa: E402
from fpl_brain import free_hit_route as fr  # noqa: E402
from fpl_brain import repositories as repo  # noqa: E402
from fpl_brain import season_rules as sr  # noqa: E402
from fpl_brain.chip_wildcard import (  # noqa: E402
    WildcardPoolBinding, WildcardPredictiveIdentity, WildcardWorldInputs,
)
from fpl_brain.database import connect_database  # noqa: E402
from fpl_brain.ingest_provenance import element_id_sha256  # noqa: E402
import free_hit_certification_fixtures as cf  # noqa: E402
from fpl_brain.models import (  # noqa: E402
    EventRecord, PickRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord,
)

ENTRY = 241392
EVENT = 5
WORLDS = 6
CAPTURED_AT = "2026-09-16T08:00:00Z"
#: Derived from the certification fixture: the authority anchors each predictive
#: dimension to the CERTIFIED BUNDLE, not to another request-owned object.
CUTOFF = cf.CUTOFF
DATA = cf.DATA_SNAPSHOT
SOURCE = cf.CODE_SNAPSHOT
CONFIG = cf.model_label(cf.MODEL_VERSIONS)
GENERATION = cf.runs_label(cf.RUNS)
CERT = cf.certification_artifact(events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3))[
    "four_gw_certification_identity"
]
#: The ONE production source of decision authority.
CERT_PATH = cf.write_artifact(
    Path(tempfile.mkdtemp()), cf.certification_artifact(events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3))
)
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
                generation=GENERATION, model_config_identity=CONFIG)
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
    """The certified decision context, derived from a REAL-schema artifact."""

    return fh.FreeHitDecisionAuthority.from_certification(
        cf.certification_artifact(
            events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3), **overrides
        ),
        loaded_from="<test certificate>",
    )


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


def _arms(*, play_scores=(1.0, 1.0, 1.0), save_scores=(1.0, 1.0, 1.0)):
    """Both arms as canonical routes from the fixture builder and the converter."""

    import free_hit_route_fixtures as fx
    from fpl_brain import free_hit_route as fr
    from fpl_brain import transfer_state as ts

    permanent = _permanent()
    universe = tuple(int(pid) for pid in UNIVERSE)
    meta = {pid: ts.PlayerMeta(pid, POSITION[pid], CLUB[pid]) for pid in universe}
    ids = tuple(sorted(OWNED))
    basis = {pid: 50 for pid in ids}
    restored_ft = fh.post_free_hit_ft_state(RULES, event_start_free_transfers=2)

    def _build(arm, events, scores, free_transfers):
        state = ts.RouteState(
            event=int(events[0]),
            players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], basis[pid]) for pid in ids),
            bank_tenths=20, free_transfers=int(free_transfers),
        )
        partial, _terminal = fx.build_canonical_route(
            start=state, events=tuple(events), meta=meta, universe=universe, price_default=50,
        )
        worlds = {
            int(event): {
                "worlds": 24, "player_ids": list(universe),
                "minutes": {pid: tuple(90.0 for _ in range(24)) for pid in universe},
                "core": {pid: tuple(float(score) for _ in range(24)) for pid in universe},
            }
            for event, score in zip(events, scores)
        }
        return fr.free_hit_route_from_canonical_route(
            partial, arm=arm, expected_events=tuple(events), rules=RULES,
            expected_start_state={
                "event": int(events[0]), "squad_ids": list(ids), "bank_tenths": 20,
                "free_transfers": int(free_transfers), "purchase_price_tenths": basis,
            },
            worlds_by_event=worlds,
            positions_of=lambda squad: {int(p): POSITION[int(p)] for p in squad},
            route_config=fx.route_config(),
        )

    return (
        _build(fr.ARM_PLAY, (EVENT + 1, EVENT + 2, EVENT + 3), play_scores, restored_ft),
        _build(fr.ARM_SAVE, (EVENT, EVENT + 1, EVENT + 2, EVENT + 3),
               (1.0, *tuple(save_scores)), 2),
    )


def _arm_inputs(*, play_scores=(1.0, 1.0, 1.0), save_scores=(1.0, 1.0, 1.0)):
    """Both arms as CANONICAL ROUTE INGREDIENTS — no matrices anywhere.

    The numeric worlds are LOADED by the adapter from the canonical world cache,
    so a test cannot supply numbers and neither can a production caller.
    """

    import free_hit_route_fixtures as fx
    from fpl_brain import transfer_state as ts

    universe = tuple(int(pid) for pid in UNIVERSE)
    meta = {pid: ts.PlayerMeta(pid, POSITION[pid], CLUB[pid]) for pid in universe}
    ids = tuple(sorted(OWNED))
    basis = {pid: 50 for pid in ids}

    def _build(events, free_transfers):
        state = ts.RouteState(
            event=int(events[0]),
            players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], basis[pid]) for pid in ids),
            bank_tenths=20, free_transfers=int(free_transfers),
        )
        partial, _terminal = fx.build_canonical_route(
            start=state, events=tuple(events), meta=meta, universe=universe, price_default=50,
        )
        return ad.FreeHitArmRoute(
            partial=partial,
            positions_of=lambda squad: {int(p): POSITION[int(p)] for p in squad},
            route_config=fx.route_config(),
        )

    return (
        _build((EVENT + 1, EVENT + 2, EVENT + 3),
               fh.post_free_hit_ft_state(RULES, event_start_free_transfers=2)),
        _build((EVENT, EVENT + 1, EVENT + 2, EVENT + 3), 2),
    )


#: A canonical world cache populated for the certified events, so the loader is
#: exercised for real rather than stubbed.
WORLD_CACHE = Path(tempfile.mkdtemp())
cf.write_world_cache(WORLD_CACHE, events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3), union=UNIVERSE)


def _permanent() -> fh.FreeHitPermanentState:
    return fh.FreeHitPermanentState(
        event=EVENT, owned_ids=tuple(sorted(OWNED)),
        purchase_price_tenths={pid: 50 for pid in OWNED}, bank_tenths=20,
        free_transfers=2, event_start_free_transfers=2, positions=POSITION, clubs=CLUB,
    )


def _certified(**overrides) -> ad.FreeHitCertifiedInputs:
    play_tail, save_tail = _arm_inputs()
    base = dict(horizon_binding=_binding(), h1_worlds=_h1_worlds(), world_identity=_identity(),
                pool_binding=_pool(), play=play_tail, save=save_tail)
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

    request = ad.build_free_hit_request(state, _certified(), conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE)
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
        ad.build_free_hit_request(state, _certified(), certification_path=CERT_PATH,
                                  world_cache_dir=WORLD_CACHE)
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
        ad.build_free_hit_request(forged, _certified(), conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE)
    assert caught.value.reasons[0] == ad.FH_CALLER_STATE_DISAGREES
    assert needle in str(caught.value), str(caught.value)


def test_a_forged_position_map_cannot_change_the_budget(conn):
    """Relabelling positions would change which temporaries are legal, and is refused."""

    _seed(conn)
    forged = _forged(conn, positions={**POSITION, 6: "MID", 16: "DEF"})
    assert forged.problems() == []          # locally it still looks like 2/5/5/3
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(forged, _certified(), conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE)
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
            conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE,
        )
    # The world no longer matches the CERTIFIED authority, not merely the
    # request's own companion object.
    assert caught.value.reasons[0] == fh.FH_DECISION_AUTHORITY_MISMATCH
    assert "cutoff" in str(caught.value)


def test_an_empty_snapshot_refuses_before_any_numeric_work(conn):
    """Every empty-snapshot route refuses, and the cause is named."""

    _seed(conn)
    state = _state(conn)
    # A mutated CERTIFICATE is how authority is attacked now; an empty snapshot
    # on the request side is caught by the contract.
    cases = (
        ("binding snapshot empty", _certified(horizon_binding=_binding(snapshot=""))),
        ("world snapshot empty",
         _certified(h1_worlds=_h1_worlds(identity=_identity(data_snapshot_sha256="")))),
    )
    # A certificate that carries no data snapshot cannot produce an authority at
    # all: the anchor is refused rather than becoming a hole in the anchoring.
    with pytest.raises(fh.FreeHitAuthorityError) as anchor:
        _authority(snapshot="")
    assert anchor.value.reasons[0] == fh.FH_DECISION_AUTHORITY_REQUIRED
    # And a certificate whose data snapshot is empty cannot be loaded at all.
    empty_cert = cf.write_artifact(
        Path(tempfile.mkdtemp()), cf.certification_artifact(
            events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3), snapshot=""), name="empty.json",
    )
    with pytest.raises(ad.FreeHitAdapterError) as loader:
        ad.build_free_hit_request(state, _certified(), conn=conn, certification_path=empty_cert, world_cache_dir=WORLD_CACHE)
    assert loader.value.reasons[0] == ad.FH_CERTIFICATION_REQUIRED
    for label, certified in cases:
        with pytest.raises(ad.FreeHitAdapterError) as caught:
            ad.build_free_hit_request(state, certified, conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE)
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
            conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE,
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
        state, _certified(h1_worlds=_h1_worlds(values)), conn=conn, certification_path=CERT_PATH,
        world_cache_dir=WORLD_CACHE,
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


# ---------------------------------------------------------------------------
# PRODUCTION CERTIFICATION AUTHORITY — exercised through the REAL loader
# ---------------------------------------------------------------------------


def _cert_file(payload=None, name="c.json"):
    return cf.write_artifact(Path(tempfile.mkdtemp()), payload, name=name)


def test_prod_A_a_genuine_certificate_authorises_the_decision(conn):
    """The full production chain: real artifact -> loader -> adapter -> request."""

    _seed(conn)
    state = _state(conn)
    request = ad.build_free_hit_request(
        state, _certified(), conn=conn, certification_path=CERT_PATH,
        world_cache_dir=WORLD_CACHE,
    )
    assert request.decision_authority is not None
    assert request.decision_authority.loaded_from.endswith("c.json") or \
        request.decision_authority.loaded_from.endswith("cert.json")
    assert request.decision_authority.certified_events == (EVENT, EVENT + 1, EVENT + 2, EVENT + 3)
    assert request.decision_authority.planning_cutoff == CUTOFF


def test_prod_B_a_different_four_event_window_refuses(conn):
    """A certificate for GW6-GW9 cannot authorise a GW5-GW8 decision."""

    _seed(conn)
    state = _state(conn)
    moved = _cert_file(cf.certification_artifact(events=(EVENT + 1, EVENT + 2, EVENT + 3, EVENT + 4)))
    with pytest.raises(ad.FreeHitAdapterError):
        ad.build_free_hit_request(state, _certified(), conn=conn, certification_path=moved)


def test_prod_C_a_later_caller_cutoff_refuses_and_pit_still_uses_the_certified_one(conn):
    """The caller cannot move the PIT replay instant."""

    _seed(conn)
    # A price captured AFTER the certified cutoff must stay invisible.
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl", started_at="2026-09-17T08:00:00Z")
        repo.insert_snapshots(conn, [
            PlayerSnapshotRecord(player_id=1, captured_at="2026-09-17T08:00:00Z", now_cost=999, raw_json={})
        ], run)
        repo.finish_fetch_run(conn, run, "success", current_event=EVENT)

    state = ad.free_hit_manager_state(conn, ENTRY, EVENT, decision_cutoff=CUTOFF)
    assert state.market_price_tenths[1] == 50, "a post-cutoff price leaked into the decision"

    later = cf.write_artifact(
        Path(tempfile.mkdtemp()),
        cf.certification_artifact(events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3),
                                  cutoff="2026-09-16T14:00:00Z"),
        name="later.json",
    )
    moved_cutoff = ad.free_hit_manager_state(
        conn, ENTRY, EVENT, decision_cutoff="2026-09-16T14:00:00Z",
    )
    forged = dataclasses.replace(_state(conn), cutoff=CUTOFF)
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(
            dataclasses.replace(forged, cutoff="2026-09-16T14:00:00Z"),
            _certified(), conn=conn, certification_path=later,
        )
    assert caught.value.reasons[0] == ad.FH_CUTOFF_MISMATCH
    assert moved_cutoff.market_price_tenths[1] == 50


def test_prod_J_a_caller_certificate_matching_itself_cannot_replace_the_persisted_one(conn):
    """Sol's decisive attack: all caller objects agree with each other.

    A caller builds a certificate B, a world identity B and a binding identity B
    that are mutually consistent, and points the adapter at B.  That is not
    authority over the PERSISTED certificate A: the loader validates B as an
    artifact, but B is a different certified context and the manager state for A's
    decision does not authorise it.
    """

    _seed(conn)
    state = _state(conn)
    world_b = _h1_worlds(identity=_identity(cutoff="2026-09-16T13:00:00Z"))
    binding_b = _binding(snapshot=DATA)
    cert_b = _cert_file(
        cf.certification_artifact(events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3),
                                  cutoff="2026-09-16T13:00:00Z"),
        name="b.json",
    )
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(
            state,
            ad.FreeHitCertifiedInputs(
                horizon_binding=binding_b, h1_worlds=world_b, world_identity=_identity(
                    cutoff="2026-09-16T13:00:00Z"),
                pool_binding=_pool(), play=_arm_inputs()[0], save=_arm_inputs()[1],
            ),
            conn=conn, certification_path=cert_b,
        )
    # The manager state is for the persisted decision, so B's cutoff does not fit.
    assert caught.value.reasons[0] in {ad.FH_CUTOFF_MISMATCH, fh.FH_DECISION_AUTHORITY_MISMATCH}


@pytest.mark.parametrize("mutation,needle", [
    ("mutated bundle", lambda: _mutated("certified_bundles")),
    ("copied digest", lambda: _copied_digest()),
    ("missing completeness audit", lambda: _without("history_completeness")),
    ("not permitted to decide", lambda: _permitted_false()),
])
def test_prod_E_an_invalid_certificate_is_refused_by_the_loader(conn, mutation, needle):
    _seed(conn)
    state = _state(conn)
    path = _cert_file(needle(), name=f"{mutation.replace(' ', '-')}.json")
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(state, _certified(), conn=conn, certification_path=path, world_cache_dir=WORLD_CACHE)
    assert caught.value.reasons[0] == ad.FH_CERTIFICATION_REQUIRED


def _mutated(key):
    payload = json.loads(json.dumps(cf.certification_artifact()))
    payload[key][str(EVENT + 1)]["runs"] = {"xpts": 999}
    return payload


def _copied_digest():
    payload = json.loads(json.dumps(cf.certification_artifact()))
    payload["planning_cutoff"] = "2026-09-16T12:30:00Z"     # component altered, digest kept
    return payload


def _without(key):
    payload = json.loads(json.dumps(cf.certification_artifact()))
    payload.pop(key)
    payload["four_gw_certification_identity"] = fg.certification_identity_of(payload)
    return payload


def _permitted_false():
    payload = json.loads(json.dumps(cf.certification_artifact()))
    payload["decision_search_permitted"] = False
    return payload

def start_state_for_save_arm(*, post_free_hit: bool = False):
    """A SAVE-arm route built from a DIFFERENT fifteen when asked.

    The SAVE arm must start H1 from the CURRENT PERMANENT squad, so a route built
    from the temporary Free Hit squad is exactly the leak the converter refuses.
    """

    import free_hit_route_fixtures as fx
    from fpl_brain import transfer_state as ts

    universe = tuple(int(pid) for pid in UNIVERSE)
    meta = {pid: ts.PlayerMeta(pid, POSITION[pid], CLUB[pid]) for pid in universe}
    # A DIFFERENT fifteen: the temporary Free Hit squad must never be a route start.
    ids = tuple(sorted(OWNED))
    if post_free_hit:
        ids = (3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 28, 29, 30)
    state = ts.RouteState(
        event=EVENT,
        players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in ids),
        bank_tenths=20, free_transfers=2,
    )
    partial, _terminal = fx.build_canonical_route(
        start=state, events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3),
        meta=meta, universe=universe, price_default=50,
    )
    return type("R", (), {"partial": partial})()


# ---------------------------------------------------------------------------
# THE PRODUCTION ROUTE BOUNDARY — Sol's three P2 findings
# ---------------------------------------------------------------------------


def test_A1_the_production_api_has_no_converted_route_parameter():
    """The direct-DTO 1,000,000 attack is impossible BY API, not merely guarded."""

    fields = set(ad.FreeHitCertifiedInputs.__dataclass_fields__)
    assert fields == {"horizon_binding", "h1_worlds", "world_identity", "pool_binding", "play", "save"}
    assert "play_route" not in fields and "save_route" not in fields
    for name in ("play", "save"):
        arm = set(ad.FreeHitArmRoute.__dataclass_fields__)
        # ONE certified world artifact per event: no parallel matrix map and no
        # parallel identity map, so the two cannot be made to disagree.
        assert arm == {"partial", "positions_of", "route_config"}
        assert not arm & {"world_identities", "worlds_by_event", "mean_net_core", "value"}
        # No route VALUE can be supplied: the arm carries ingredients, not results.
        assert not arm & {"mean_net_core", "events", "route_value", "evaluation"}


def test_A2_a_canonical_partial_route_is_converted_internally(conn):
    """The adapter calls the converter, so only converted routes reach the request."""

    _seed(conn)
    state = _state(conn)
    request = ad.build_free_hit_request(state, _certified(), conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE)
    assert request.play_route.arm == fr.ARM_PLAY
    assert request.save_route.arm == fr.ARM_SAVE
    assert [e.event for e in request.play_route.events] == [EVENT + 1, EVENT + 2, EVENT + 3]
    assert [e.event for e in request.save_route.events] == [EVENT, EVENT + 1, EVENT + 2, EVENT + 3]
    # Both were valued by the canonical evaluator, not by the caller.
    assert all(e.mean_gross_core > 0.0 for e in request.save_route.events)


def test_A3_the_save_route_starts_from_the_canonical_permanent_h1_state(conn):
    _seed(conn)
    request = ad.build_free_hit_request(
        _state(conn), _certified(), conn=conn, certification_path=CERT_PATH,
        world_cache_dir=WORLD_CACHE,
    )
    assert request.save_route.start_state["squad_ids"] == sorted(OWNED)
    assert request.save_route.start_state["event"] == EVENT
    assert request.save_route.start_state["free_transfers"] == 2


def test_A4_the_play_route_never_starts_from_a_temporary_squad(conn):
    _seed(conn)
    request = ad.build_free_hit_request(
        _state(conn), _certified(), conn=conn, certification_path=CERT_PATH,
        world_cache_dir=WORLD_CACHE,
    )
    assert request.play_route.start_state["squad_ids"] == sorted(OWNED)
    assert request.play_route.start_state["event"] == EVENT + 1
    assert request.play_route.start_state["free_transfers"] == fh.post_free_hit_ft_state(
        RULES, event_start_free_transfers=2
    )


def test_A5_a_route_with_the_wrong_start_state_refuses(conn):
    """The converter's basis/state authority still applies through production."""

    _seed(conn)
    play, save = _arm_inputs()
    # The SAVE route must start H1 from the CURRENT PERMANENT squad; building it
    # from a different fifteen is the leak the converter refuses.
    wrong = start_state_for_save_arm(post_free_hit=True)
    broken = ad.FreeHitArmRoute(
        partial=wrong.partial, positions_of=save.positions_of, route_config=save.route_config,
    )
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(
            _state(conn),
            ad.FreeHitCertifiedInputs(
                horizon_binding=_binding(), h1_worlds=_h1_worlds(), world_identity=_identity(),
                pool_binding=_pool(), play=play, save=broken,
            ),
            conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE,
        )
    assert caught.value.reasons[0] == fh.FH_TAIL_ROUTE_INVALID


def test_B_matrix_authority_has_no_caller_surface_at_all():
    """The decisive answer: identity-A / matrix-B is UNREPRESENTABLE.

    At the reviewed base SHA a caller could build a provenance-carrying world
    artifact whose identity was the genuine certified H2 identity and whose numeric
    matrix was arbitrary.  There is now no matrix parameter on the production path:
    the worlds are LOADED from the canonical authority, so the numbers the
    evaluator consumes are never caller-supplied.
    """

    import inspect

    arm_fields = set(ad.FreeHitArmRoute.__dataclass_fields__)
    assert arm_fields == {"partial", "positions_of", "route_config"}
    assert not arm_fields & {"certified_worlds_by_event", "worlds_by_event", "world_identities",
                             "matrix", "core", "worlds"}
    parameters = set(inspect.signature(ad.build_free_hit_request).parameters)
    assert not parameters & {"worlds_by_event", "certified_worlds_by_event", "world_identities",
                             "matrix", "core", "values", "mean_net_core"}
    # There is also no way to hand the loader a matrix: it reads the canonical cache.
    assert "world_cache_dir" in parameters


def test_B2_an_absent_cache_regenerates_from_the_certified_runs(conn):
    """A cache MISS does not weaken authority: the loader regenerates from the
    certified run ids, so the matrix is derived from the certification either way.
    """

    _seed(conn)
    empty_cache = Path(tempfile.mkdtemp())
    request = ad.build_free_hit_request(
        _state(conn), _certified(), conn=conn, certification_path=CERT_PATH,
        world_cache_dir=empty_cache,
    )
    from fpl_brain import route_optimizer as ro
    for event in (EVENT, EVENT + 1, EVENT + 2, EVENT + 3):
        assert (empty_cache / f"{ro.world_cache_key(
            event=event, bundle=ad._CertifiedRunIds(event, cf.RUNS),
            config=ro.OptimizerConfig(policy_selection_worlds=12),
            union_ids=tuple(int(p) for p in UNIVERSE),
        )}.json").exists(), f"event {event} was not materialised under its certified key"
    # The regenerated worlds are canonical, not the fixture's scores.
    assert len(request.save_route.events) == 4
    assert all(e.mean_net_core == e.mean_net_core for e in request.save_route.events)


def test_B3_a_cache_for_another_universe_is_simply_never_read(conn):
    """The key includes the capture union, so foreign numbers cannot be substituted.

    A cache written for a different universe lands under a DIFFERENT key and is
    never read for this decision: the matrix comes from the certified key or from
    regeneration, never from a caller's file.
    """

    _seed(conn)
    foreign = Path(tempfile.mkdtemp())
    cf.write_world_cache(foreign, events=(EVENT, EVENT + 1, EVENT + 2, EVENT + 3), union=UNIVERSE[:5])
    before = {p.name for p in foreign.iterdir()}
    request = ad.build_free_hit_request(
        _state(conn), _certified(), conn=conn, certification_path=CERT_PATH,
        world_cache_dir=foreign,
    )
    # The foreign files were never the ones consumed: the loader wrote its OWN
    # certified matrices alongside them.
    after = {p.name for p in foreign.iterdir()}
    assert len(after) == len(before) + 4, "the loader did not materialise the four certified keys"
    assert len(request.save_route.events) == 4


def test_B4_the_loaded_matrix_is_the_canonical_one_for_the_certified_bundle(conn):
    """The matrix the request carries IS the loader's output for that event."""

    _seed(conn)
    request = ad.build_free_hit_request(
        _state(conn), _certified(), conn=conn, certification_path=CERT_PATH,
        world_cache_dir=WORLD_CACHE,
    )
    from fpl_brain import route_optimizer as ro
    expected_key = cf.write_world_cache(
        Path(tempfile.mkdtemp()), events=(EVENT,), union=UNIVERSE,
    )[EVENT]
    # The value loaded is the one written under the canonical key for the certified
    # bundle: a caller-supplied matrix could never have produced it.
    assert expected_key == ro.world_cache_key(
        event=EVENT, bundle=ad._CertifiedRunIds(EVENT, cf.RUNS),
        config=request.save_route.events and ro.OptimizerConfig(policy_selection_worlds=12),
        union_ids=tuple(int(p) for p in UNIVERSE),
    )
    assert request.save_route.value() > 0.0


# ---------------------------------------------------------------------------
# C — CURRENT FREE TRANSFERS ARE MANAGER AUTHORITY TOO
# ---------------------------------------------------------------------------


def test_C1_a_forged_current_free_transfers_refuses(conn):
    """Canonical current FT = 2, caller current FT = 5, everything else canonical."""

    _seed(conn)
    canonical = _state(conn)
    forged = dataclasses.replace(canonical, free_transfers=5)
    assert forged.event_start_free_transfers == canonical.event_start_free_transfers
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(forged, _certified(), conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE)
    assert caught.value.reasons[0] == ad.FH_CALLER_STATE_DISAGREES
    assert "current free transfers" in str(caught.value)


def test_C2_a_forged_event_start_free_transfers_also_refuses(conn):
    """The two concepts stay INDEPENDENT: neither is inferred from the other."""

    _seed(conn)
    canonical = _state(conn)
    forged = dataclasses.replace(canonical, event_start_free_transfers=5)
    assert forged.free_transfers == canonical.free_transfers
    with pytest.raises(ad.FreeHitAdapterError) as caught:
        ad.build_free_hit_request(forged, _certified(), conn=conn, certification_path=CERT_PATH, world_cache_dir=WORLD_CACHE)
    assert caught.value.reasons[0] == ad.FH_CALLER_STATE_DISAGREES
    assert "event-start free transfers" in str(caught.value)
