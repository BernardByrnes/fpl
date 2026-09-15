"""Production construction path: adapter -> request -> evaluator -> arbiter.

The integration test begins from authoritative production-STYLE inputs (manager
state, official generation, canonical prices and bases, certified prediction
evidence, world inputs, a real four-GW route, chip definitions) and never
hand-assembles a ``WildcardRequest``.  That is the whole point of the adapter:
a production caller supplies authorities, not internal field values.
"""

from __future__ import annotations

import pytest

from fpl_brain import chip_decision as cd
from fpl_brain import chip_wildcard as wc
from fpl_brain import manager_lineup
from fpl_brain import season_rules as sr
from fpl_brain import transfer_state as ts
from fpl_brain import wildcard_request_adapter as ad
from fpl_brain.ingest_provenance import element_id_sha256

RULES = sr.SeasonRules(season="2026/27")
CUTOFF = "2026-09-12T19:20:00Z"
IDENTITY = "sha256:" + "c" * 64
SNAPSHOT = "sha256:" + "d" * 64
SOURCE = "sha256:" + "5" * 64
GENERATION = "generation-2026-09-12T19:20:00Z"
CONFIG_ID = "sha256:" + "7" * 64
PLANNING_EVENT = 5

CLUBS = tuple(range(1, 21))


def _identity(**over):
    """The complete predictive identity every artifact in this fixture carries."""

    base = dict(cutoff=CUTOFF, data_snapshot_sha256=SNAPSHOT, source_snapshot_sha256=SOURCE,
                generation=GENERATION, model_config_identity=CONFIG_ID)
    base.update(over)
    return wc.WildcardPredictiveIdentity(**base)


def _pool(n_gkp=4, n_def=12, n_mid=14, n_fwd=10, points=3.0, price=50):
    """A production-shaped universe with per-row certified identity."""

    players: dict[int, wc.WildcardPlayer] = {}
    pid = 1
    for position, count in (("GKP", n_gkp), ("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)):
        for index in range(count):
            events = {}
            for event in range(PLANNING_EVENT, PLANNING_EVENT + 8):
                events[event] = wc.WildcardPlayerEvent(
                    event=event, expected_points=points + 0.3 * (index % 3),
                    expected_minutes=90.0, p_start=0.8, availability=1.0,
                    fixture_count=1, identity=_identity(),
                )
            players[pid] = wc.WildcardPlayer(
                player_id=pid, position=position, club_id=CLUBS[pid % len(CLUBS)],
                market_price_tenths=price, events=events, web_name=f"p{pid}",
            )
            pid += 1
    return players


def _worlds(players, *, events=range(PLANNING_EVENT, PLANNING_EVENT + 8)):
    ids = tuple(sorted(players))
    worlds = {}
    for event in events:
        minutes, core = {}, {}
        for pid in ids:
            entry = players[pid].at(event)
            minutes[pid] = [0.0 if entry is None else float(entry.expected_minutes)]
            core[pid] = [0.0 if entry is None else float(entry.expected_points)]
        worlds[int(event)] = wc.WildcardWorldInputs(
            event=int(event),
            worlds=1, player_ids=ids, minutes=minutes, core=core, identity=_identity()
        )
    return worlds


def _legal_owned(players) -> list[int]:
    chosen, clubs = [], {}
    for position, required in ts.POSITION_COMPOSITION.items():
        picked = 0
        for pid, player in sorted(players.items()):
            if player.position != position or picked >= required:
                continue
            if clubs.get(player.club_id, 0) >= ts.SQUAD_TEAM_LIMIT:
                continue
            chosen.append(pid)
            clubs[player.club_id] = clubs.get(player.club_id, 0) + 1
            picked += 1
    assert len(chosen) == ts.SQUAD_SIZE
    return chosen


def _manager(players, owned, **over):
    base = dict(
        entry_id=241392,
        planning_event=PLANNING_EVENT,
        squad_ids=tuple(sorted(owned)),
        bank_tenths=30,
        purchase_price_tenths={int(p): players[int(p)].market_price_tenths for p in owned},
        event_start_free_transfers=2,
        chip_availability=tuple(_chip_rows(PLANNING_EVENT)),
        market_price_tenths={int(pid): int(pl.market_price_tenths) for pid, pl in players.items()},
    )
    base.update(over)
    return ad.WildcardManagerState(**base)


def _chip_rows(planning_event, *, windows=((2, 19), (20, 38))):
    rows = []
    for start, stop in windows:
        in_window = start <= planning_event <= stop
        rows.append({
            "name": "wildcard", "number": 1, "chip_type": "wildcard",
            "window_start_event": start, "window_stop_event": stop,
            "window": f"GW{start}-GW{stop}", "used": False, "used_event": None,
            "available_for_event": in_window, "expired": planning_event > stop,
        })
    return rows


def _chip_binding(planning_event=PLANNING_EVENT):
    return cd.ChipHorizonBinding(
        planning_event=planning_event,
        horizon_events=cd.canonical_chip_horizon(planning_event),
        certification_identity=IDENTITY, data_snapshot_sha256=SNAPSHOT,
    )


def _certified(players, *, generation=None, players_override=None, worlds_override=None):
    horizon = wc.wildcard_horizon(PLANNING_EVENT, length=8)
    ids = tuple(sorted(players))
    return ad.WildcardCertifiedInputs(
        horizon=horizon,
        chip_horizon_binding=_chip_binding(),
        value_horizon_binding=wc.value_horizon_binding(
            PLANNING_EVENT, horizon, decision_cutoff=CUTOFF, data_snapshot_sha256=SNAPSHOT,
            source_snapshot_sha256=SOURCE, prediction_generation=GENERATION,
            model_config_identity=CONFIG_ID,
        ),
        players=players_override if players_override is not None else players,
        worlds_by_event=worlds_override if worlds_override is not None else _worlds(players),
        generation=generation if generation is not None else {
            "accepted": True, "captured_at": GENERATION, "official_element_count": len(ids),
            "element_ids": list(ids), "element_id_sha256": element_id_sha256(list(ids)),
        },
    )


def _route(players, owned, *, mean_net_core=40.0, terminal_bank_tenths=30, terminal_ft=1):
    entries = []
    for event in range(PLANNING_EVENT, PLANNING_EVENT + 4):
        entries.append(wc.WildcardSaveRouteEvent(
            event=event, squad_ids=tuple(sorted(owned)), bank_tenths=30,
            purchase_price_tenths={int(p): players[int(p)].market_price_tenths for p in owned},
            free_transfers=1, mean_net_core=float(mean_net_core), hit_points=0,
        ))
    return wc.WildcardSaveRoute(
        events=tuple(entries), terminal_squad_ids=tuple(sorted(owned)),
        terminal_bank_tenths=int(terminal_bank_tenths),
        terminal_purchase_price_tenths=entries[-1].purchase_price_tenths,
        terminal_free_transfers=int(terminal_ft), cumulative_hits=0, wildcard_available=True,
    )


def _build(players, owned, **over):
    return ad.build_wildcard_request(
        _manager(players, owned, **{k: v for k, v in over.items() if k in ("bank_tenths", "event_start_free_transfers", "purchase_price_tenths")}),
        _certified(players, **{k: v for k, v in over.items() if k in ("generation", "players_override", "worlds_override")}),
        over.get("route", _route(players, owned)),
        rules=RULES, data_snapshot_sha256=SNAPSHOT,
        reservation=over.get("reservation"),
    )


# ---------------------------------------------------------------------------
# §11 — the production-style integration test
# ---------------------------------------------------------------------------


def test_adapter_A_production_construction_path_end_to_end():
    """authoritative state -> adapter -> request -> evaluator -> arbiter."""

    players = _pool()
    owned = _legal_owned(players)
    request = _build(players, owned)

    # the adapter produced a real WildcardRequest, no internal field fabricated
    assert isinstance(request, wc.WildcardRequest)
    assert request.owned_ids == tuple(sorted(owned))
    assert request.pool_binding is not None
    assert request.value_horizon_binding is not None
    assert request.save_route is not None

    # canonical selling values survived construction
    for pid in owned:
        assert request.selling_price_tenths[int(pid)] == ts.selling_price_tenths(
            players[int(pid)].market_price_tenths, players[int(pid)].market_price_tenths)

    evaluation = wc.evaluate_wildcard(request)
    assert evaluation.candidate_metrics["mean_paired_uplift"] is not None
    assert evaluation.evidence["official_player_pool"]["complete"] is True
    assert evaluation.evidence["save_policy"]["route_value_evidence"]["manual_hit_subtraction"] is False
    assert evaluation.evidence["executable"] is False

    # --- the arbiter, with a reservation spy
    class Spy:
        calls = 0
        payload = None

        def estimate(self, *, action, planning_event, expiry_event, state):
            Spy.calls += 1
            Spy.payload = dict(state)
            return cd.ReservationEstimate(
                value=1.0, calibration_status=cd.CALIBRATION_CALIBRATED,
                terminal_value=0.0, weeks_to_expiry=8, reason_codes=(),
                conditional_on=("weeks_remaining",),
            )

    decision = cd.decide_chip_action(
        horizon_binding=_chip_binding(),
        chip_availability=_chip_rows(PLANNING_EVENT),
        evaluations={cd.CHIP_ACTION_WC: evaluation},
        reservation=Spy(),
        certification_valid=True,
        manager_state={"squad_ids": list(owned)},
    )
    assert Spy.calls == 1
    assert decision.status != cd.STATUS_PLAY_CHIP
    assert decision.status in (cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED, cd.STATUS_CHIP_REVIEW_REQUIRED)
    # the reservation received the REAL post-SAVE state
    assert Spy.payload["squad_ids"] == list(sorted(owned))
    assert "bank_tenths" in Spy.payload


def test_adapter_B_the_adapter_never_subtracts_route_hits():
    players = _pool()
    owned = _legal_owned(players)
    request = _build(players, owned, route=_route(players, owned, mean_net_core=100.0))
    evaluation = wc.evaluate_wildcard(request)
    evidence = evaluation.evidence["save_policy"]["route_value_evidence"]
    assert evidence["manual_hit_subtraction"] is False
    assert evidence["hit_authority"] == "ROUTE_MEAN_NET_CORE_ALREADY_NETS_HITS"


# ---------------------------------------------------------------------------
# §12 — adapter fail-closed matrix
# ---------------------------------------------------------------------------


def test_adapter_C_missing_manager_state_refuses():
    players = _pool()
    owned = _legal_owned(players)
    broken = _manager(players, owned, squad_ids=())
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(broken, _certified(players), _route(players, owned),
                                  rules=RULES, data_snapshot_sha256=SNAPSHOT)
    assert ad.WC_MANAGER_STATE_MISSING in str(exc.value)


def test_adapter_D_missing_official_generation_refuses():
    players = _pool()
    owned = _legal_owned(players)
    with pytest.raises(wc.WildcardInputError):
        _build(players, owned, generation={})


def test_adapter_E_an_eligible_player_without_a_projection_refuses():
    players = _pool()
    owned = _legal_owned(players)
    reduced = {pid: p for pid, p in players.items() if pid != 3}
    with pytest.raises(ad.WildcardAdapterError) as exc:
        _build(players, owned, players_override=reduced)
    assert "OFFICIAL_PLAYER_POOL_INCOMPLETE" in str(exc.value)


def test_adapter_F_missing_acquisition_basis_refuses():
    players = _pool()
    owned = _legal_owned(players)
    broken = _manager(players, owned, purchase_price_tenths={})
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(broken, _certified(players), _route(players, owned),
                                  rules=RULES, data_snapshot_sha256=SNAPSHOT)
    assert "acquisition basis" in str(exc.value)


def test_adapter_G_a_cached_selling_value_that_disagrees_refuses():
    players = _pool()
    owned = _legal_owned(players)
    wrong = {int(p): players[int(p)].market_price_tenths + 30 for p in owned}
    broken = _manager(players, owned, cached_selling_price_tenths=wrong)
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(broken, _certified(players), _route(players, owned),
                                  rules=RULES, data_snapshot_sha256=SNAPSHOT)
    assert "selling value" in str(exc.value)


def test_adapter_H_a_disagreeing_chip_binding_refuses():
    players = _pool()
    owned = _legal_owned(players)
    certified = _certified(players)
    mismatched = ad.WildcardCertifiedInputs(
        horizon=certified.horizon,
        chip_horizon_binding=cd.ChipHorizonBinding(
            planning_event=PLANNING_EVENT + 1,
            horizon_events=cd.canonical_chip_horizon(PLANNING_EVENT + 1),
            certification_identity=IDENTITY, data_snapshot_sha256=SNAPSHOT),
        value_horizon_binding=certified.value_horizon_binding,
        players=certified.players, worlds_by_event=certified.worlds_by_event,
        generation=certified.generation,
    )
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(_manager(players, owned), mismatched, _route(players, owned),
                                  rules=RULES, data_snapshot_sha256=SNAPSHOT)
    assert "planning event" in str(exc.value)


def test_adapter_I_incomplete_world_inputs_refuse_at_the_evaluator():
    players = _pool()
    owned = _legal_owned(players)
    partial = dict(_worlds(players))
    partial.pop(9)  # one horizon event has no worlds
    request = _build(players, owned, worlds_override=partial)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_WORLD_INPUTS_MISSING in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_adapter_J_missing_route_refuses():
    players = _pool()
    owned = _legal_owned(players)
    evaluation = wc.evaluate_wildcard(_build(players, owned, route=None))
    assert wc.WC_SAVE_ROUTE_MISSING in evaluation.reason_codes


def test_adapter_K_route_missing_a_required_event_refuses():
    players = _pool()
    owned = _legal_owned(players)
    route = _route(players, owned)
    gapped = wc.WildcardSaveRoute(
        events=tuple(e for e in route.events if e.event != 6),
        terminal_squad_ids=route.terminal_squad_ids,
        terminal_bank_tenths=route.terminal_bank_tenths,
        terminal_purchase_price_tenths=route.terminal_purchase_price_tenths,
        terminal_free_transfers=route.terminal_free_transfers,
    )
    evaluation = wc.evaluate_wildcard(_build(players, owned, route=gapped))
    assert wc.WC_SAVE_ROUTE_INVALID in evaluation.reason_codes


def test_adapter_L_route_with_a_nonfinite_net_value_refuses():
    players = _pool()
    owned = _legal_owned(players)
    route = _route(players, owned)
    broken = wc.WildcardSaveRoute(
        events=tuple(wc.WildcardSaveRouteEvent(
            event=e.event, squad_ids=e.squad_ids, bank_tenths=e.bank_tenths,
            purchase_price_tenths=e.purchase_price_tenths, free_transfers=e.free_transfers,
            mean_net_core=float("nan") if e.event == 6 else e.mean_net_core,
        ) for e in route.events),
        terminal_squad_ids=route.terminal_squad_ids,
        terminal_bank_tenths=route.terminal_bank_tenths,
        terminal_purchase_price_tenths=route.terminal_purchase_price_tenths,
        terminal_free_transfers=route.terminal_free_transfers,
    )
    evaluation = wc.evaluate_wildcard(_build(players, owned, route=broken))
    assert wc.WC_SAVE_ROUTE_INVALID in evaluation.reason_codes


def test_adapter_M_unresolved_terminal_ft_refuses():
    players = _pool()
    owned = _legal_owned(players)
    route = _route(players, owned, terminal_ft=-1)
    evaluation = wc.evaluate_wildcard(_build(players, owned, route=route))
    assert wc.WC_SAVE_ROUTE_INVALID in evaluation.reason_codes


def test_adapter_N_no_active_wc_definition_refuses():
    players = _pool()
    owned = _legal_owned(players)
    request = _build(players, owned)
    request = wc.WildcardRequest(**{**request.__dict__, "chip_availability": ()})
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_WINDOW_UNRESOLVED in evaluation.reason_codes


def test_adapter_O_overlapping_wc_definitions_refuse():
    players = _pool()
    owned = _legal_owned(players)
    broken = _manager(players, owned,
                      chip_availability=tuple(_chip_rows(PLANNING_EVENT, windows=((2, 19), (4, 30)))))
    request = ad.build_wildcard_request(broken, _certified(players), _route(players, owned),
                                        rules=RULES, data_snapshot_sha256=SNAPSHOT)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_WINDOW_AMBIGUOUS in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


# ---------------------------------------------------------------------------
# §4 — the REAL production path: adapter -> converter -> exact_evaluate -> request
# ---------------------------------------------------------------------------


def _canonical_route_fixture(players, owned):
    """A genuine canonical PartialRoute built from apply_transfer_batch.

    Mirrors the route engine's own construction: one canonical transition per
    event, each carrying before_state / squad_after / next_event_state.
    """

    from fpl_brain import route_optimizer as ro
    from fpl_brain import transfer_state as ts

    meta = {int(pid): ts.PlayerMeta(player_id=int(pid), position=players[int(pid)].position,
                                    club_id=int(players[int(pid)].club_id))
            for pid in owned}
    price = {int(pid): int(players[int(pid)].market_price_tenths) for pid in owned}
    state = ts.RouteState(
        event=PLANNING_EVENT,
        players=tuple(ts.RoutePlayer(player_id=p, position=meta[p].position,
                                     club_id=meta[p].club_id, purchase_price_tenths=price[p])
                      for p in sorted(owned)),
        bank_tenths=30, free_transfers=1, event_start_free_transfers=1,
    )
    actions = []
    for event in range(PLANNING_EVENT, PLANNING_EVENT + 4):
        snapshot = ts.PriceSnapshot(event=int(event), prices=price)
        transition = ts.apply_transfer_batch(state, ts.TransferBatch.roll(), snapshot, meta)
        nxt = transition.next_event_state
        actions.append({
            "event": int(event), "kind": "ROLL", "batch": ts.TransferBatch.roll(),
            "transition": transition, "hit_points": 0, "delta_3gw": 0.0,
            "squad_ids": tuple(sorted(int(p.player_id) for p in nxt.players)),
            "ft_after": int(nxt.free_transfers), "bank_after": int(nxt.bank_tenths),
        })
        state = nxt

    class _Partial:
        def __init__(self):
            self.actions = tuple(actions)
            self.state = state
            self.hits = 0

    return _Partial()


def _route_worlds(partial, players):
    from fpl_brain import route_optimizer as ro

    worlds = {}
    for event, squad in ro.route_event_squads(partial):
        ids = tuple(sorted(int(p) for p in squad))
        worlds[int(event)] = {
            "worlds": 1, "player_ids": list(ids),
            "core": {pid: [1.0] for pid in ids},
            "minutes": {pid: [90.0] for pid in ids},
        }
    return worlds


def _route_positions(players):
    def positions_of(squad_ids):
        return {int(p): players[int(p)].position for p in squad_ids if int(p) in players}

    return positions_of


def test_adapter_P_production_happy_path_reaches_exact_evaluate():
    """THE regression for the stale call site.

    A valid canonical route plus authoritative evaluation inputs must flow
    build_wildcard_request -> converter -> exact_evaluate -> a valid
    WildcardRequest, with no refusal and no TypeError, and WITHOUT any
    caller-supplied evaluation result.
    """

    import inspect

    from fpl_brain import route_optimizer as ro

    players = _pool()
    owned = _legal_owned(players)

    # there is no evaluation-RESULT parameter on the production adapter any more
    parameters = inspect.signature(ad.build_wildcard_request).parameters
    assert "route_evaluation" not in parameters
    assert "route_worlds_by_event" in parameters

    partial = _canonical_route_fixture(players, owned)
    request = ad.build_wildcard_request(
        _manager(players, owned), _certified(players), None,
        rules=RULES, data_snapshot_sha256=SNAPSHOT,
        canonical_route=partial,
        route_worlds_by_event=_route_worlds(partial, players),
        route_positions_of=_route_positions(players),
        route_config=ro.OptimizerConfig(),
        route_events=tuple(range(PLANNING_EVENT, PLANNING_EVENT + 4)),
    )

    assert isinstance(request, wc.WildcardRequest)
    assert request.save_route is not None
    assert len(request.save_route.events) == 4
    # the values are the canonical exact evaluation of THIS route
    evaluation = ro.exact_evaluate(
        partial, worlds_by_event=_route_worlds(partial, players),
        positions_of=_route_positions(players), cache={},
        config=ro.OptimizerConfig(), events=tuple(range(PLANNING_EVENT, PLANNING_EVENT + 4)),
    )
    expected = {int(r["event"]): float(r["mean_net_core"]) for r in evaluation["per_event"]}
    assert [e.mean_net_core for e in request.save_route.events] == \
           [expected[e.event] for e in request.save_route.events]

    # and the full decision still runs, review-only
    decision = wc.evaluate_wildcard(request)
    assert decision.candidate_metrics["mean_paired_uplift"] is not None or decision.reason_codes


def test_adapter_Q_no_caller_supplied_evaluation_result_exists():
    """There is no production argument representing caller-authoritative values."""

    import inspect

    for name in ("route_evaluation", "evaluation", "route_values", "mean_net_core"):
        assert name not in inspect.signature(ad.build_wildcard_request).parameters, name
    assert "evaluation" not in inspect.signature(
        ad.wildcard_save_route.wildcard_save_route_from_canonical_route
    ).parameters


def test_adapter_R_missing_authoritative_inputs_refuses():
    """Without the authoritative inputs the adapter refuses rather than inventing."""

    players = _pool()
    owned = _legal_owned(players)
    partial = _canonical_route_fixture(players, owned)
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(
            _manager(players, owned), _certified(players), None,
            rules=RULES, data_snapshot_sha256=SNAPSHOT, canonical_route=partial,
        )
    assert "authoritative evaluation INPUTS" in str(exc.value)
