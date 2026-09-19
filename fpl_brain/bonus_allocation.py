"""PE-3 — exact FPL bonus allocation from fixture-level BPS totals.

Bonus is a FIXTURE-LEVEL COMPETITION, not a per-player quantity: the officials rank every
player in the match by BPS and award bonus by rank.  This module is the one canonical
implementation of that allocation.  It is a pure function of the BPS totals — no player
identity, no position, no minutes, no prediction logic, and no database access.

OFFICIAL RULE (Premier League / FPL, as published):

    a group of players on the same BPS receives   3 - (number of players with a
    STRICTLY HIGHER BPS), provided that value is greater than zero.

Two consequences that matter and that this module is written to honour:

  * TIED PLAYERS ALL RECEIVE THE SAME BONUS, and the players below them are pushed down
    by the whole tied group.  There is deliberately no player-id or any other arbitrary
    tie-breaker: a tie is shared.
  * THE TOTAL BONUS AWARDED IN A FIXTURE CAN EXCEED SIX.  It is six only when the top
    three BPS totals are distinct.  Any code that asserts ``sum(bonus) == 6`` is wrong.

Verified against official data: fixture 19 of the 2026/27 season has BPS 29, 29, 28, 27, …
and official bonus 3, 3, 1, 0, … — seven bonus points in total, which this function
reproduces exactly.  See ``scripts/replay_bonus_allocation.py`` for the full replay over
every eligible FINAL fixture.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

#: Identity of the allocation semantics.  Bump only if the OFFICIAL rule changes.
BONUS_ALLOCATOR_VERSION = "fpl_bonus_allocation_v1.0.0"

#: The highest bonus a single player can receive.
MAX_BONUS = 3


class BonusAllocationError(ValueError):
    """The supplied BPS totals cannot be ranked."""


def _finite(value: Any, player_id: int) -> float:
    """Normalise one BPS total to a finite float WITHOUT truncating or rounding.

    Official BPS arrive as integers and must replay identically; a predictive proxy is a
    CONTINUOUS score whose fractional part carries real ordering information, so
    ``int(...)`` would invent ties (30.9 and 30.1 would both become 30) and change the
    allocation.  Comparison is therefore numeric on the supplied value: equal values share
    bonus, strictly greater ranks higher, and there is no epsilon anywhere.
    """

    try:
        number = float(value)
    except (TypeError, ValueError) as failure:
        raise BonusAllocationError(f"non-numeric BPS {value!r} for player {player_id}") from failure
    if not math.isfinite(number):
        raise BonusAllocationError(f"non-finite BPS {value!r} for player {player_id}")
    return number


def allocate_fixture_bonus(bps_by_player: Mapping[int, int]) -> dict[int, int]:
    """Allocate bonus for ONE fixture from that fixture's official player BPS totals.

    ``bps_by_player`` maps player id -> that player's BPS in this fixture.  The mapping
    must be the COMPLETE set of players eligible in the fixture: the ranking is a
    competition, so an omitted high-BPS player would change everyone else's bonus.  The
    caller is responsible for completeness; this function cannot detect a missing player
    and does not guess.

    Returns player id -> bonus (0, 1, 2 or 3).  Every supplied player appears in the
    result, including those awarded 0.

    Invariants, in the terms the phase requires:

    * equal BPS  =>  equal bonus;
    * a strictly higher BPS can never receive LESS bonus than a strictly lower one;
    * bonus is always one of 0, 1, 2, 3;
    * no arbitrary tie-breaker of any kind (the result is invariant under permutation of
      the input, and depends only on the multiset of BPS values);
    * deterministic.
    """

    if not bps_by_player:
        return {}

    values = {int(player_id): _finite(bps, int(player_id))
              for player_id, bps in bps_by_player.items()}

    # Descending BPS.  The sort key is the VALUE ALONE: players on an exactly equal value
    # are grouped below, so their relative order here can never affect the result.
    ordered = sorted(values.items(), key=lambda item: -item[1])

    # For each distinct value, the index of its FIRST occurrence in the descending order is
    # exactly the number of players with a strictly greater BPS.
    first_index_by_value: dict[float, int] = {}
    for index, (_player_id, bps) in enumerate(ordered):
        first_index_by_value.setdefault(bps, index)

    bonus_by_player: dict[int, int] = {}
    for player_id, bps in ordered:
        bonus = MAX_BONUS - first_index_by_value[bps]
        bonus_by_player[player_id] = bonus if bonus > 0 else 0
    return bonus_by_player


def allocation_is_consistent(bps_by_player: Mapping[int, int],
                             bonus_by_player: Mapping[int, int]) -> tuple[bool, str]:
    """Check a bonus allocation against the official rule, for the replay gate.

    Returns ``(ok, detail)``.  Used to verify an allocation that came from somewhere else
    (an artifact, a database row set) rather than from :func:`allocate_fixture_bonus`.
    """

    expected = allocate_fixture_bonus(bps_by_player)
    supplied = {int(player_id): int(bonus) for player_id, bonus in bonus_by_player.items()}
    if set(supplied) != set(int(p) for p in bps_by_player):
        return False, "player sets differ between BPS totals and bonus"
    mismatches = [
        (player_id, expected[player_id], supplied[player_id])
        for player_id in sorted(expected)
        if supplied[player_id] != expected[player_id]
    ]
    if mismatches:
        return False, f"{len(mismatches)} mismatches, e.g. {mismatches[:5]}"
    return True, f"{len(expected)} players match"


def bonus_totals(bonus_by_player: Mapping[int, int]) -> dict[str, Any]:
    """Diagnostic summary of one fixture's allocation (never used as a rule)."""

    total = sum(int(value) for value in bonus_by_player.values())
    return {
        "players": len(bonus_by_player),
        "total_bonus": total,
        "exceeds_six": total > 6,
        "awarded": sorted(int(p) for p, b in bonus_by_player.items() if int(b) > 0),
    }
