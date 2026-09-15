"""FINDING D — the canonical SAVE-route authority and its one-way converter.

The converter must prove a route came from the accepted normal-transfer engine.
These tests build GENUINE canonical transitions with
``transfer_state.apply_transfer_batch`` rather than hand-rolled lookalikes, so
the continuity and terminal checks are exercised against real route semantics.
"""

from __future__ import annotations

import pytest

from fpl_brain import chip_wildcard as wc
from fpl_brain import season_rules as sr
from fpl_brain import transfer_state as ts
from fpl_brain import wildcard_save_route as wsr

RULES = sr.SeasonRules(season="2026/27")
PLANNING_EVENT = 5
EVENTS = (5, 6, 7, 8)


PRICE = 50


def _Meta(player_id: int, position: str, club_id: int, price: int = PRICE):
    """The CANONICAL PlayerMeta the accepted transition consumes."""

    return ts.PlayerMeta(player_id=int(player_id), position=position, club_id=int(club_id))


def _squad_meta():
    """A legal 15: 2 GKP / 5 DEF / 5 MID / 3 FWD, <=3 per club."""

    meta: dict[int, _Meta] = {}
    pid = 1
    clubs = tuple(range(1, 21))
    for position, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        for _ in range(count):
            meta[pid] = _Meta(pid, position, clubs[pid % len(clubs)])
            pid += 1
    return meta


def _state(meta, event):
    return ts.RouteState(
        event=int(event),
        players=tuple(ts.RoutePlayer(player_id=p.player_id, position=p.position,
                                     club_id=p.club_id, purchase_price_tenths=PRICE)
                      for p in meta.values()),
        bank_tenths=20,
        free_transfers=1,
        event_start_free_transfers=1,
    )


def _canonical_route(meta, *, events=EVENTS):
    """A genuine canonical route: one ROLL transition per event."""

    state = _state(meta, events[0])
    actions = []
    for event in events:
        snapshot = ts.PriceSnapshot(event=int(event), prices={p.player_id: PRICE for p in meta.values()})
        transition = ts.apply_transfer_batch(state, ts.TransferBatch.roll(), snapshot, meta)
        next_state = transition.next_event_state
        action = {
            "event": int(event), "kind": "ROLL", "batch": ts.TransferBatch.roll(),
            "transition": transition, "hit_points": 0, "delta_3gw": 0.0,
            "squad_ids": tuple(sorted(int(p.player_id) for p in next_state.players)),
            "ft_after": int(next_state.free_transfers),
            "bank_after": int(next_state.bank_tenths),
        }
        actions.append(action)
        state = next_state
    return _Partial(actions=tuple(actions), state=state, hits=0)


class _Partial:
    def __init__(self, *, actions, state, hits):
        self.actions = actions
        self.state = state
        self.hits = hits


def _evaluation(net=100.0, *, events=EVENTS, over=None):
    over = over or {}
    return {"per_event": [
        {"event": e, "kind": "ROLL", "hit_points": 0,
         "mean_gross_core": over.get(e, net), "mean_net_core": over.get(e, net)}
        for e in events
    ]}


def _horizon():
    return wc.wildcard_horizon(PLANNING_EVENT, length=8)


def _convert(partial, evaluation, **over):
    kwargs = dict(planning_event=PLANNING_EVENT, horizon=_horizon(), rules=RULES,
                  expected_horizon=EVENTS)
    kwargs.update(over)
    return wsr.wildcard_save_route_from_canonical_route(partial, evaluation, **kwargs)


# ---------------------------------------------------------------------------
# A — canonical conversion
# ---------------------------------------------------------------------------


def test_D_A_a_valid_canonical_route_converts():
    meta = _squad_meta()
    route = _convert(_canonical_route(meta), _evaluation())
    assert len(route.events) == 4
    assert tuple(e.event for e in route.events) == EVENTS
    assert route.problems() == []
    # the values came from the evaluation, not from the caller
    assert all(e.mean_net_core == 100.0 for e in route.events)
    # the terminal state is the canonical final state
    canonical = _canonical_route(meta)
    assert route.terminal_squad_ids == tuple(sorted(int(p.player_id) for p in canonical.state.players))
    assert route.terminal_bank_tenths == int(canonical.state.bank_tenths)


def test_D_A2_the_converted_route_feeds_a_wildcard_request():
    """The converter's output is usable as the SAVE input."""

    meta = _squad_meta()
    route = _convert(_canonical_route(meta), _evaluation())
    assert isinstance(route, wc.WildcardSaveRoute)
    assert route.wildcard_available is True


# ---------------------------------------------------------------------------
# L — arbitrary mean_net_core cannot enter
# ---------------------------------------------------------------------------


def test_D_L_a_route_without_its_exact_evaluation_refuses():
    meta = _squad_meta()
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_canonical_route(meta), {"per_event": []})
    assert "exact-evaluation value" in str(exc.value)


def test_D_L2_a_nonfinite_evaluation_value_refuses():
    meta = _squad_meta()
    with pytest.raises(wsr.WildcardSaveRouteError):
        _convert(_canonical_route(meta), _evaluation(over={6: float("nan")}))


# ---------------------------------------------------------------------------
# C/D/E/F — squad legality
# ---------------------------------------------------------------------------


def _tamper_event_squad(partial, event, players):
    """Replace one event's resulting squad with an EXPLICIT player tuple."""

    actions = []
    for action in partial.actions:
        if int(action["event"]) != int(event):
            actions.append(action)
            continue
        next_state = action["transition"].next_event_state
        tampered = ts.RouteState(
            event=int(next_state.event),
            players=tuple(players),
            bank_tenths=int(next_state.bank_tenths), free_transfers=int(next_state.free_transfers),
            event_start_free_transfers=next_state.event_start_free_transfers,
        )
        actions.append({**action, "transition": _Transition(tampered)})
        last_tampered = int(action["event"]) == int(partial.actions[-1]["event"])
    # Tampering a NON-final event must not silently move the terminal state, or
    # the terminal-equality check would fire instead of the intended one.
    return _Partial(actions=tuple(actions),
                    state=tampered if last_tampered else partial.state,
                    hits=partial.hits)


def _players_for(meta, ids, *, club_over=None, basis=PRICE):
    club_over = club_over or {}
    return tuple(
        ts.RoutePlayer(player_id=int(pid), position=meta[pid].position,
                       club_id=int(club_over.get(pid, meta[pid].club_id)),
                       purchase_price_tenths=basis)
        for pid in ids
    )


class _Transition:
    """A tampered transition that still exposes the canonical shapes."""

    def __init__(self, next_state):
        self.next_event_state = next_state
        self.squad_after = next_state
        self.before_state = None


def test_D_C_a_short_event_squad_refuses():
    meta = _squad_meta()
    partial = _canonical_route(meta)
    short = sorted(meta)[:-1]
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_tamper_event_squad(partial, 6, _players_for(meta, short)), _evaluation())
    assert "expected 15" in str(exc.value)


def test_D_D_a_duplicate_player_refuses():
    meta = _squad_meta()
    partial = _canonical_route(meta)
    ids = sorted(meta)
    dup = ids[:14] + [ids[0]]             # 15 slots, exactly one id repeated
    assert len(dup) == 15 and len(set(dup)) == 14
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_tamper_event_squad(partial, 6, _players_for(meta, dup)), _evaluation())
    assert "duplicate" in str(exc.value)


def test_D_E_an_illegal_position_composition_refuses():
    meta = _squad_meta()
    partial = _canonical_route(meta)
    ids = sorted(meta)
    gkp = [p for p in ids if meta[p].position == "GKP"]
    deff = [p for p in ids if meta[p].position == "DEF"]
    mid = [p for p in ids if meta[p].position == "MID"]
    fwd = [p for p in ids if meta[p].position == "FWD"]
    # The canonical pool holds exactly one legal composition, so an extra MID is
    # added to the fixture: the 15-man set below is UNIQUE and 15-strong but has
    # MID 6 / FWD 2, which is illegal.
    extra = 16
    meta[extra] = _Meta(extra, "MID", 19)
    mixed = gkp[:2] + deff[:5] + mid[:5] + [extra] + fwd[:2]
    assert len(mixed) == 15 and len(set(mixed)) == 15
    assert len(mixed) == 15 and len(set(mixed)) == 15
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_tamper_event_squad(partial, 6, _players_for(meta, mixed)), _evaluation())
    assert "need" in str(exc.value)



def test_D_F_exceeding_the_club_limit_refuses():
    meta = _squad_meta()
    # put four players at one club, keeping the composition legal
    partial = _canonical_route(meta)
    ids = sorted(meta)
    over_club = {pid: 1 for pid in ids[:4]}    # four players at club 1
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_tamper_event_squad(partial, 6, _players_for(meta, ids, club_over=over_club)),
                 _evaluation())
    assert "club limit" in str(exc.value)


# ---------------------------------------------------------------------------
# G/H — continuity and terminal equality
# ---------------------------------------------------------------------------


def test_D_G_a_discontinuous_route_refuses():
    meta = _squad_meta()
    partial = _canonical_route(meta)
    # give event 7 a before_state that does NOT match event 6's result
    actions = []
    for action in partial.actions:
        if int(action["event"]) != 7:
            actions.append(action)
            continue
        transition = action["transition"]
        base = _state(meta, 99)
        foreign = ts.RouteState(event=99, players=base.players, bank_tenths=999,
                                free_transfers=1, event_start_free_transfers=1)
        actions.append({**action, "transition": _TransitionWithBefore(transition.next_event_state, foreign)})
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_Partial(actions=tuple(actions), state=partial.state, hits=partial.hits), _evaluation())
    assert "does not continue" in str(exc.value)


class _TransitionWithBefore:
    def __init__(self, next_state, before_state):
        self.next_event_state = next_state
        self.squad_after = next_state
        self.before_state = before_state


def test_D_H_a_terminal_state_that_differs_from_h4_refuses():
    meta = _squad_meta()
    partial = _canonical_route(meta)
    lookalike = _state(meta, 99)
    lookalike = ts.RouteState(event=99, players=lookalike.players, bank_tenths=999, free_transfers=1,
                              event_start_free_transfers=1)
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_Partial(actions=partial.actions, state=lookalike, hits=partial.hits), _evaluation())
    assert "terminal state is not its final event state" in str(exc.value)


# ---------------------------------------------------------------------------
# I — acquisition basis tampering
# ---------------------------------------------------------------------------


def test_D_I_a_tampered_acquisition_basis_refuses():
    """Squad membership looks plausible but one basis is wrong.

    The basis is read from the canonical state, so a tampered basis is caught by
    the terminal/H4 equality check: the state carrying the wrong basis cannot
    simultaneously be the transition's result and the route's terminal state.
    """

    meta = _squad_meta()
    partial = _canonical_route(meta)
    next_state = partial.actions[-1]["transition"].next_event_state
    tampered_players = tuple(
        ts.RoutePlayer(player_id=p.player_id, position=p.position, club_id=p.club_id,
                       purchase_price_tenths=int(p.purchase_price_tenths) + 7)
        for p in next_state.players
    )
    tampered = ts.RouteState(event=int(next_state.event), players=tampered_players,
                             bank_tenths=int(next_state.bank_tenths),
                             free_transfers=int(next_state.free_transfers),
                             event_start_free_transfers=next_state.event_start_free_transfers)
    with pytest.raises(wsr.WildcardSaveRouteError):
        _convert(_Partial(actions=partial.actions, state=tampered, hits=partial.hits), _evaluation())


# ---------------------------------------------------------------------------
# J/K — FT bounds and identity
# ---------------------------------------------------------------------------


def test_D_J_free_transfers_above_the_canonical_maximum_refuse():
    meta = _squad_meta()
    partial = _canonical_route(meta)
    next_state = partial.actions[0]["transition"].next_event_state
    over_ft = ts.RouteState(event=int(next_state.event), players=next_state.players,
                            bank_tenths=int(next_state.bank_tenths),
                            free_transfers=int(RULES.max_free_transfers) + 1,
                            event_start_free_transfers=1)
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(_tamper_event_squad(partial, 5, sorted(meta)) if False else
                 _Partial(actions=({**partial.actions[0], "transition": _Transition(over_ft)},) + partial.actions[1:],
                          state=partial.state, hits=0), _evaluation())
    assert "outside" in str(exc.value)


def test_D_K_a_route_over_the_wrong_horizon_refuses():
    meta = _squad_meta()
    partial = _canonical_route(meta)
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(partial, _evaluation(), expected_horizon=(5, 6, 7, 9))
    assert "normal horizon" in str(exc.value)


def test_D_K2_a_route_with_the_wrong_event_count_refuses():
    meta = _squad_meta()
    partial = _canonical_route(meta, events=(5, 6, 7))
    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(partial, _evaluation(events=(5, 6, 7)))
    assert "exactly 4 events" in str(exc.value)


# ---------------------------------------------------------------------------
# B — production cannot self-certify a synthetic SAVE route
# ---------------------------------------------------------------------------


def test_D_B_the_adapter_refuses_a_synthetic_route_in_production(tmp_path):
    """A caller-built WildcardSaveRoute is not authority when a live connection
    is present."""

    from fpl_brain import database
    from fpl_brain import wildcard_request_adapter as ad
    from tests.test_wildcard_request_adapter import (
        CONFIG_ID, CUTOFF, GENERATION, IDENTITY, PLANNING_EVENT as PE, SOURCE,
        _certified, _chip_rows, _legal_owned, _manager, _pool, _route,
    )

    players = _pool()
    owned = _legal_owned(players)
    conn = database.connect_database(str(tmp_path / "prod.db"))
    # a recorded ACCEPTED generation, so the pool check passes and the SAVE-route
    # authority check is the thing under test
    from fpl_brain import repositories as repo
    from fpl_brain.ingest_provenance import element_id_sha256
    ids = sorted(players)
    repo.record_bootstrap_generation(
        conn, captured_at=GENERATION, accepted=True, official_element_count=len(ids),
        parsed_count=len(ids), persisted_count=len(ids), element_ids=ids,
        element_ids_sha256=element_id_sha256(ids), acceptance_rule="test",
        acceptance_rule_version="v1", club_player_counts={}, availability_counts={},
    )
    conn.commit()
    synthetic = _route(players, owned)
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(
            _manager(players, owned), _certified(players), synthetic,
            rules=RULES, data_snapshot_sha256="sha256:" + "d" * 64, conn=conn,
        )
    assert "WILDCARD_SAVE_ROUTE_MISSING" in str(exc.value)
    conn.close()


def test_D_B2_the_adapter_requires_the_routes_exact_evaluation_when_canonical():
    from fpl_brain import wildcard_request_adapter as ad
    from tests.test_wildcard_request_adapter import (
        _certified, _legal_owned, _manager, _pool,
    )

    players = _pool()
    owned = _legal_owned(players)
    meta = _squad_meta()
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(
            _manager(players, owned), _certified(players), None,
            rules=RULES, data_snapshot_sha256="sha256:" + "d" * 64,
            canonical_route=_canonical_route(meta), route_evaluation=None,
        )
    assert "exact-evaluation result" in str(exc.value)


# ---------------------------------------------------------------------------
# §8 — the fabricated-route attack on the PRODUCTION path
# ---------------------------------------------------------------------------


def test_D_fabricated_route_attack_refuses():
    """A caller-built "route" with one-player squads, arbitrary finite
    mean_net_core, plausible bank/FT and matching event numbers must NOT become
    SAVE authority.  The converter proves legality and continuity from canonical
    transitions, so a fabricated blob has nothing that satisfies them.
    """

    meta = _squad_meta()
    ids = sorted(meta)

    def fake_state(squad):
        return ts.RouteState(
            event=6, players=tuple(ts.RoutePlayer(player_id=p, position=meta[p].position,
                                                  club_id=meta[p].club_id, purchase_price_tenths=99)
                                   for p in squad),
            bank_tenths=999, free_transfers=3, event_start_free_transfers=3,
        )

    class FakeTransition:
        """Structurally plausible, canonically impossible."""

        def __init__(self, state):
            self.next_event_state = state
            self.squad_after = state
            self.before_state = None

    fake_actions = tuple(
        # one-player squads, matching event numbers, arbitrary finite value
        {"event": e, "kind": "ROLL", "transition": FakeTransition(fake_state([ids[0]])),
         "hit_points": 0, "squad_after": (), "bank_after": 999}
        for e in EVENTS
    )
    fake_route = _Partial(actions=fake_actions, state=fake_state([ids[0]]), hits=0)
    evaluation = _evaluation(net=1_000_000.0)      # arbitrary finite value

    with pytest.raises(wsr.WildcardSaveRouteError) as exc:
        _convert(fake_route, evaluation)
    assert "expected 15" in str(exc.value)

    # and no numeric SAVE authority can be produced from it
    route_attempt = None
    try:
        route_attempt = _convert(fake_route, evaluation)
    except wsr.WildcardSaveRouteError:
        pass
    assert route_attempt is None
