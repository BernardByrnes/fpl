"""Finding C — the official player pool must not be caller-self-certified.

Two defects are closed here:

  * ``pool_binding_from_generation`` accepted a caller-supplied mapping/digest and
    treated a MISSING ``accepted`` flag as acceptance, so a caller could
    self-certify a universe.
  * ``screen_players`` iterated every supplied player while ``pool_accounting``
    only reasoned about the BOUND ids, so an extra non-official player (the "999"
    counterexample) could be screened, optimised and selected while the
    accounting still reported complete.
"""

from __future__ import annotations

import pytest

from fpl_brain import chip_wildcard as wc
from fpl_brain import database
from fpl_brain import repositories as repo
from fpl_brain import wildcard_request_adapter as ad
from fpl_brain.ingest_provenance import element_id_sha256
from tests.test_wildcard_request_adapter import (
    CUTOFF, GENERATION, _certified, _chip_rows, _legal_owned, _manager, _pool, _route,
)

RULES = ad.sr.SeasonRules(season="2026/27")
SNAPSHOT = "sha256:" + "d" * 64


def _generation(ids, *, accepted=True, include_flag=True, digest=None):
    row = {
        "captured_at": GENERATION,
        "official_element_count": len(ids),
        "element_ids": list(ids),
        "element_id_sha256": digest if digest is not None else element_id_sha256(sorted(ids)),
    }
    if include_flag:
        row["accepted"] = accepted
    return row


def _universe_with_extra(players, extra_id=999):
    """The canonical universe plus a tempting non-official player 999."""

    augmented = dict(players)
    events = {}
    # Finding B gives rows their own COMPLETE predictive identity; this fixture
    # (added by Finding C, before B) is adapted mechanically to that field.
    reference = next(iter(players[min(players)].events.values()))
    for event in players[min(players)].events:
        events[event] = wc.WildcardPlayerEvent(
            event=event, expected_points=99.0, expected_minutes=90.0, p_start=1.0,
            availability=1.0, fixture_count=1, identity=reference.identity,
        )
    augmented[extra_id] = wc.WildcardPlayer(
        player_id=extra_id, position="FWD", club_id=1, market_price_tenths=40,
        events=events, web_name="intruder",
    )
    return augmented


# ---------------------------------------------------------------------------
# §9 — generation acceptance
# ---------------------------------------------------------------------------


def test_pool_A_an_explicitly_accepted_generation_may_proceed():
    players = _pool()
    owned = _legal_owned(players)
    request = ad.build_wildcard_request(
        _manager(players, owned), _certified(players), _route(players, owned),
        rules=RULES, data_snapshot_sha256=SNAPSHOT,
    )
    assert request.pool_binding is not None
    assert wc.pool_accounting(request)["complete"] is True


def test_pool_B_a_missing_accepted_status_refuses():
    """Absence of an acceptance flag is NOT acceptance."""

    ids = tuple(range(1, 4))
    with pytest.raises(wc.WildcardInputError) as exc:
        wc.pool_binding_from_generation(_generation(ids, include_flag=False))
    assert wc.OFFICIAL_PLAYER_POOL_INCOMPLETE in str(exc.value)


def test_pool_C_an_explicitly_rejected_generation_refuses():
    ids = tuple(range(1, 4))
    with pytest.raises(wc.WildcardInputError) as exc:
        wc.pool_binding_from_generation(_generation(ids, accepted=False))
    assert wc.OFFICIAL_PLAYER_POOL_INCOMPLETE in str(exc.value)


def test_pool_D_a_generation_digest_that_disagrees_refuses():
    ids = tuple(range(1, 4))
    with pytest.raises(wc.WildcardInputError):
        wc.pool_binding_from_generation(_generation(ids, digest="sha256:" + "0" * 64))


def test_pool_E_the_store_is_the_authority(tmp_path):
    """pool_binding_from_store resolves from the canonical accepted record."""

    conn = database.connect_database(str(tmp_path / "gen.db"))
    ids = [1, 2, 3, 4]
    repo.record_bootstrap_generation(
        conn,
        captured_at=GENERATION, accepted=True, official_element_count=len(ids),
        parsed_count=len(ids), persisted_count=len(ids),
        element_ids=ids, element_ids_sha256=element_id_sha256(ids),
        acceptance_rule="test", acceptance_rule_version="v1",
        club_player_counts={}, availability_counts={},
    )
    conn.commit()
    binding = wc.pool_binding_from_store(conn)
    assert binding.eligible_ids == tuple(ids)
    assert binding.generation_id_sha256 == element_id_sha256(ids)
    conn.close()


# ---------------------------------------------------------------------------
# §6/§8 — universe reconciliation
# ---------------------------------------------------------------------------


def test_pool_F_the_999_counterexample_is_refused_and_never_screened():
    """THE decisive regression.

    The canonical pool is the request's own universe; player 999 is added with
    excellent projections and complete world inputs -- everything the optimizer
    would want.  The request must refuse, and 999 must never reach the screen,
    the frontier, the improvement universe or the selected squad.
    """

    players = _pool()
    owned = _legal_owned(players)
    canonical_ids = tuple(sorted(players))
    augmented = _universe_with_extra(players)
    assert 999 in augmented

    certified = ad.WildcardCertifiedInputs(
        horizon=_certified(players).horizon,
        chip_horizon_binding=_certified(players).chip_horizon_binding,
        value_horizon_binding=_certified(players).value_horizon_binding,
        players=augmented,
        worlds_by_event=_worlds_for(augmented),
        generation=_generation(canonical_ids),
    )
    request = ad.build_wildcard_request(
        _manager(augmented, owned), certified, _route(augmented, owned),
        rules=RULES, data_snapshot_sha256=SNAPSHOT,
    )
    # the request is constructed, but the evaluator must refuse it
    evaluation = wc.evaluate_wildcard(request)
    assert wc.OFFICIAL_PLAYER_POOL_INCOMPLETE in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None

    accounting = wc.pool_accounting(request)
    assert accounting["complete"] is False
    assert 999 in accounting["extra_ids"]

    # and even if screening were reached, it iterates the CANONICAL universe only
    scores, stats = wc.screen_players(request)
    assert 999 not in scores
    assert stats["screened"] == len(canonical_ids)

    candidates, frontier_stats = wc.build_wildcard_candidates(request, scores)
    for candidate in candidates:
        assert 999 not in candidate.squad


def test_pool_G_a_same_count_wrong_id_set_refuses():
    players = _pool()
    owned = _legal_owned(players)
    shifted = {pid + 1: player for pid, player in players.items()}
    for new_id, player in shifted.items():
        shifted[new_id] = wc.WildcardPlayer(
            player_id=new_id, position=player.position, club_id=player.club_id,
            market_price_tenths=player.market_price_tenths, events=dict(player.events),
            web_name=player.web_name,
        )
    certified = ad.WildcardCertifiedInputs(
        horizon=_certified(players).horizon,
        chip_horizon_binding=_certified(players).chip_horizon_binding,
        value_horizon_binding=_certified(players).value_horizon_binding,
        players=shifted,
        worlds_by_event=_worlds_for(shifted),
        generation=_generation(tuple(sorted(players))),
    )
    shifted_owned = [pid + 1 for pid in owned]
    # The same COUNT with a different ID SET must fail.  The adapter refuses at
    # its universe check (a canonical eligible id has no projection row) with the
    # canonical token; if it ever reached accounting, the same universe would be
    # reported as both unaccounted ids and extras.
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(
            _manager(shifted, shifted_owned), certified, _route(shifted, shifted_owned),
            rules=RULES, data_snapshot_sha256=SNAPSHOT,
        )
    assert "OFFICIAL_PLAYER_POOL_INCOMPLETE" in str(exc.value)


def test_pool_H_a_missing_official_player_refuses_as_unaccounted_not_excluded():
    players = _pool()
    owned = _legal_owned(players)
    canonical = tuple(sorted(players))
    dropped = canonical[-1]
    reduced = {pid: p for pid, p in players.items() if pid != dropped}
    certified = ad.WildcardCertifiedInputs(
        horizon=_certified(players).horizon,
        chip_horizon_binding=_certified(players).chip_horizon_binding,
        value_horizon_binding=_certified(players).value_horizon_binding,
        players=reduced,
        worlds_by_event=_worlds_for(reduced),
        generation=_generation(canonical),
    )
    # The ADAPTER refuses first, with the canonical token, because a canonical
    # eligible player has no projection row at all.
    with pytest.raises(ad.WildcardAdapterError) as exc:
        ad.build_wildcard_request(
            _manager(reduced, owned), certified, _route(reduced, owned),
            rules=RULES, data_snapshot_sha256=SNAPSHOT,
        )
    assert "OFFICIAL_PLAYER_POOL_INCOMPLETE" in str(exc.value)

    # And at the evaluator level the absent player is classified UNACCOUNTED,
    # never as an exclusion -- that distinction is the point.
    reduced_owned = [pid for pid in owned if pid != dropped]
    raw = wc.WildcardRequest(
        planning_event=5, horizon=_certified(players).horizon,
        players=reduced,
        positions={pid: p.position for pid, p in reduced.items()},
        owned_ids=tuple(sorted(reduced_owned)),
        purchase_price_tenths={int(p): players[int(p)].market_price_tenths for p in reduced_owned},
        selling_price_tenths={int(p): players[int(p)].market_price_tenths for p in reduced_owned},
        bank_tenths=30, rules=RULES,
        horizon_binding=_certified(players).chip_horizon_binding,
        certification_identity="sha256:" + "c" * 64,
        data_snapshot_sha256=SNAPSHOT,
        event_start_free_transfers=2,
        chip_availability=tuple(_chip_rows(5)),
        value_horizon_binding=_certified(players).value_horizon_binding,
        worlds_by_event=_worlds_for(reduced),
        pool_binding=wc.pool_binding_from_generation(_generation(canonical)),
    )
    accounting = wc.pool_accounting(raw)
    assert accounting["complete"] is False
    assert dropped in accounting["unaccounted_ids"], "absent is NOT excluded"
    assert dropped not in accounting["excluded_ids"]


# ---------------------------------------------------------------------------
# §10 — supported / excluded accounting
# ---------------------------------------------------------------------------


def test_pool_I_supported_and_excluded_partition_the_eligible_pool():
    players = _pool()
    owned = _legal_owned(players)
    canonical = tuple(sorted(players))
    # strip a few players' horizon support so they are legitimately EXCLUDED
    stripped = dict(players)
    for pid in canonical[:3]:
        short = {e: v for e, v in players[pid].events.items() if e < 10}
        stripped[pid] = wc.WildcardPlayer(pid, players[pid].position, players[pid].club_id,
                                          players[pid].market_price_tenths, short, "short")
    certified = ad.WildcardCertifiedInputs(
        horizon=_certified(players).horizon,
        chip_horizon_binding=_certified(players).chip_horizon_binding,
        value_horizon_binding=_certified(players).value_horizon_binding,
        players=stripped,
        worlds_by_event=_worlds_for(stripped),
        generation=_generation(canonical),
    )
    request = ad.build_wildcard_request(
        _manager(stripped, owned), certified, _route(stripped, owned),
        rules=RULES, data_snapshot_sha256=SNAPSHOT,
    )
    accounting = wc.pool_accounting(request)
    supported, excluded, eligible = (
        set(accounting["supported_ids"]), set(accounting["excluded_ids"]), set(canonical)
    )
    assert supported | excluded == eligible
    assert supported & excluded == set()
    assert accounting["screened_count"] == len(supported)
    scores, _ = wc.screen_players(request)
    assert set(scores) == supported, "screened ids must equal the supported ids exactly"
    # full machine-readable exclusion audit, not a truncated sample
    assert set(accounting["excluded_reasons"]) == excluded


def _worlds_for(players, events=range(5, 13)):
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
            worlds=1, player_ids=ids, minutes=minutes, core=core
        )
    return worlds
