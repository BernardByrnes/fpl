"""The canonical normal-route contract, traced and PINNED before use.

PHASE 4 of the Wildcard wiring consumes the accepted normal four-GW route for
the SAVE arm.  Before subtracting anything, the contract had to be traced, and
the answer to the decisive question is recorded here so the double-count cannot
be reintroduced by a later change:

    DOES ROUTE VALUE ALREADY INCLUDE TRANSFER HITS?  ->  YES.

``route_optimizer.exact_evaluate`` returns ``net_core = gross_core -
cumulative_hits`` and every per-event entry carries BOTH ``mean_gross_core`` and
``mean_net_core``, plus its own ``hit_points``.  There is therefore exactly ONE
hit authority: a consumer that evaluates against ``net_core`` (or per-event
``mean_net_core``) must subtract NOTHING further.  Consuming ``gross_core`` and
subtracting ``cumulative_hits`` once is equally valid; doing both is a
double-count and is what these tests exist to prevent.

The remaining traced answers:

  * each event exposes its resulting squad -- ``route_event_squads(partial)``
    gives ``[(event, squad_ids)]`` and each per-event entry carries the full
    lineup policy
  * bank            -- ``RouteState.bank_tenths``; the transition exposes
                       ``bank_before_tenths`` / ``bank_after_tenths``
  * acquisition basis -- ``RoutePlayer.purchase_price_tenths`` on every player
  * FT state        -- ``RouteState.free_transfers`` (available for the current
                       event) and ``event_start_free_transfers`` (the START of
                       event bank, which a same-event Wildcard must preserve)
  * before or after that event's transfers -- BOTH: ``squad_after`` is the state
    after this event's transfers and keeps the CURRENT event's start FT value;
    ``next_event_state`` is the state carried INTO the next event with the
    rolled-over FT
  * the authoritative terminal state after H4 is the route's final
    ``PartialRoute.state`` (equivalently the last ``next_event_state``)

These are assertions about EXISTING canonical code; nothing here is modified.
"""

from __future__ import annotations

import inspect

import pytest

from fpl_brain import route_optimizer as ro
from fpl_brain import transfer_state as ts


def test_the_route_value_contract_exposes_hits_exactly_once():
    """net_core = gross_core - cumulative_hits, with per-event gross AND net."""

    source = inspect.getsource(ro.exact_evaluate)
    assert '"cumulative_hits": hits, "net_core": gross - hits' in source, (
        "the route value contract changed: hits may no longer be deducted exactly "
        "once, which is the double-count this test pins"
    )
    assert '"mean_gross_core": float(entry["mean_gross"]), "mean_net_core": float(entry["mean_gross"]) - hit' in source
    assert '"hit_points": hit' in source


def test_the_route_exposes_the_per_event_squad_and_policy():
    source = inspect.getsource(ro.route_event_squads)
    assert 'a["squad_ids"]' in source
    evaluate_source = inspect.getsource(ro.exact_evaluate)
    for key in ("starter_ids", "bench_gk_id", "bench_outfield_order", "captain_id", "vice_captain_id"):
        assert key in evaluate_source, key


def test_the_route_state_exposes_bank_basis_and_ft():
    fields = set(ts.RouteState.__dataclass_fields__)
    for name in ("bank_tenths", "free_transfers", "event_start_free_transfers",
                 "cumulative_hit_points", "players", "transfers_made_this_event"):
        assert name in fields, name
    player_fields = set(ts.RoutePlayer.__dataclass_fields__)
    assert "purchase_price_tenths" in player_fields, "acquisition basis must be on the route player"


def test_the_transition_exposes_resulting_state_on_both_sides_of_the_event():
    source = inspect.getsource(ts.apply_transfer_batch)
    for key in ("squad_after", "next_event_state", "bank_after_tenths", "hit_points"):
        assert key in source, key
    # squad_after keeps the CURRENT event's start FT (needed for same-event
    # Wildcard preservation); next_event_state carries the rolled-over FT.
    assert "event_start_free_transfers=state.event_start_free_transfers" in source
    assert "event_start_free_transfers=next_ft" in source
    # cumulative hits accumulate once, from the transition's own hit_points
    assert "int(state.cumulative_hit_points) + hit_points" in source


def test_hit_cost_is_the_canonical_constant():
    assert ts.HIT_POINTS_PER_EXTRA_TRANSFER == 4
    free_used, paid, points = ts.hit_points_for(ft_before=1, transfer_count=2)
    assert (free_used, paid, points) == (1, 1, 4), "one paid transfer must cost exactly 4"


def test_a_consumer_of_net_core_must_not_subtract_hits_again():
    """The arithmetic the SAVE arm must follow, stated as a worked example.

    A route whose gross is 100 with one paid transfer is worth 96 net.  Reading
    ``net_core`` and subtracting 4 again would give 92; reading ``gross_core``
    and subtracting 4 gives 96.  Both are one deduction; only one of them is a
    single deduction if the field already nets.
    """

    gross, hits = 100.0, 4
    assert gross - hits == 96.0
    net_core = gross - hits
    assert net_core == 96.0
    # the mistake this pins:
    assert net_core - hits == 92.0 != 96.0
