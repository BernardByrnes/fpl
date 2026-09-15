"""Production construction path for ``BenchBoostRequest``.

This module is the SEAM between accepted production state and the Bench Boost
evaluator.  It assembles already-accepted contracts; it creates no Bench Boost
architecture of its own.

    canonical PlanningContext squad  ─┐
    canonical player positions        │
    captured admissible lineup        ├─► build_bench_boost_request ─► BenchBoostRequest
    canonical chip availability       │                                     │
    certified four-GW horizon+worlds ─┘                                     ▼
                                                                    evaluate_bench_boost
                                                                            │
                                                                            ▼
                                                                     chip arbiter

Two deliberate boundaries:

  * ``bench_boost_manager_state(conn, ...)`` is the ONLY part that touches the
    database, and it reads canonical accessors read-only.
  * ``build_bench_boost_request(...)`` is assembly plus verification.  It never
    substitutes a value it could not source: a missing lineup, a lineup that no
    longer describes the canonical fifteen, or an illegal fifteen refuses rather
    than being defaulted.

THE SQUAD IS NOT A CALLER ARGUMENT
----------------------------------
The fifteen and their positions come from ``manager_worlds.resolve_squad`` --
the same accepted authority the manager-world builder uses -- not from a list a
caller supplies.  When the caller passes the connection the state was derived
from, ``build_bench_boost_request`` RE-DERIVES the canonical state and refuses
on any disagreement, so a hand-made squad cannot be laundered into a chip
decision while the canonical manager state exists.

THE LINEUP IS THE MANAGER'S OWN
-------------------------------
Bench Boost is not a lineup optimizer.  The XI, bench GK, bench order and
armband come from the manager's captured admissible lineup (``squad_picks``),
which is the only record of what the manager actually fielded.  If that lineup
names a different fifteen than the canonical squad, it is stale and the adapter
refuses rather than guessing a replacement fifteen -- a fabricated lineup would
silently change the very counterfactual the chip is measured against.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import chip_bench_boost as bb
from . import chip_decision as cd
from . import manager_lineup as ml
from . import manager_worlds as mw
from . import planning as planning_module
from . import repositories as repo
from . import transfer_state as ts

BENCH_BOOST_ADAPTER_VERSION = "bench_boost_adapter_v1.0.0"

BB_MANAGER_STATE_MISSING = "BENCH_BOOST_PRODUCTION_MANAGER_STATE_MISSING"
BB_LINEUP_NOT_CAPTURED = "BENCH_BOOST_ADMISSIBLE_LINEUP_NOT_CAPTURED"
BB_LINEUP_SQUAD_MISMATCH = "BENCH_BOOST_CAPTURED_LINEUP_IS_NOT_THE_CANONICAL_SQUAD"
BB_LINEUP_ILLEGAL = "BENCH_BOOST_CAPTURED_LINEUP_IS_NOT_LEGAL"
BB_CALLER_STATE_DISAGREES = "BENCH_BOOST_CALLER_STATE_DISAGREES_WITH_CANONICAL"
#: A production request must be built against canonical manager authority.  A
#: caller holding only a hand-made state has no authority to assert positions,
#: clubs or a lineup, so it must say so explicitly.
BB_CANONICAL_AUTHORITY_REQUIRED = "BENCH_BOOST_CANONICAL_AUTHORITY_REQUIRED"
BB_HORIZON_MISMATCH = "BENCH_BOOST_CERTIFIED_HORIZON_MISMATCH"


class BenchBoostAdapterError(cd.ChipInputError):
    """The production adapter could not assemble an authoritative request."""


# ---------------------------------------------------------------------------
# Authoritative manager state (the ONLY database-touching part)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchBoostManagerState:
    """The manager facts the adapter needs, sourced canonically.

    ``positions`` and ``clubs`` are CANONICAL: they come from the player
    authority (``route_comparator.load_player_meta``), never from a caller's
    label map.  A caller that relabels one canonical player's position while
    keeping the same fifteen would change which autosubs are legal and therefore
    the chip's value, so ``build_bench_boost_request`` re-derives both and
    refuses on any disagreement.
    """

    entry_id: int
    planning_event: int
    squad_ids: tuple[int, ...]
    policy: ml.ManagerPolicy
    positions: Mapping[int, str]
    clubs: Mapping[int, int]
    chip_availability: tuple[Mapping[str, Any], ...]
    #: The event whose captured lineup supplied the XI/bench/armband.  Recorded
    #: so the report can show WHICH submitted lineup was evaluated.
    lineup_source_event: int
    #: True when the lineup came from the planning event itself rather than the
    #: most recent earlier capture.
    lineup_is_exact_event: bool = False

    def problems(self) -> list[str]:
        found: list[str] = []
        # The FULL fifteen, not merely a fieldable eleven: exact composition and
        # the club limit are properties of the squad itself.
        found.extend(ts.squad_composition_errors(self.squad_ids, self.positions, self.clubs))
        legality = ml.policy_legality_errors(self.policy, self.positions)
        if legality:
            found.append(f"lineup is not legal: {legality}")
        if {int(pid) for pid in self.squad_ids} != set(bb.squad_ids_of(self.policy)):
            found.append("the lineup's players are not the canonical squad")
        if not self.chip_availability:
            found.append("no chip availability rows")
        return found


def _canonical_positions_and_clubs(
    conn: sqlite3.Connection, squad_ids: Sequence[int]
) -> tuple[dict[int, str], dict[int, int]]:
    """Canonical position and club per player, from the accepted player authority.

    ``route_comparator.load_player_meta`` is the SAME reader the accepted route
    layer uses (``scoring_rules.POSITION_IDS`` over ``players.element_type``),
    and it is the only place the club is available at all -- the manager-world
    resolver discards it.  A squad member the player table does not describe is
    reported rather than defaulted, so an unknown label can never satisfy a
    composition count.
    """

    from . import route_comparator as rc

    meta = rc.load_player_meta(conn, squad_ids)
    missing = sorted(int(pid) for pid in squad_ids if int(pid) not in meta)
    if missing:
        raise BenchBoostAdapterError(
            f"{BB_MANAGER_STATE_MISSING}: the player authority does not describe squad player(s) {missing[:8]}",
            reasons=(BB_MANAGER_STATE_MISSING,),
        )
    positions = {int(pid): str(meta[int(pid)].position) for pid in squad_ids}
    clubs = {int(pid): int(meta[int(pid)].club_id) for pid in squad_ids}
    return positions, clubs


def _lineup_policy(
    rows: Sequence[Mapping[str, Any]], positions: Mapping[int, str]
) -> tuple[ml.ManagerPolicy | None, str | None]:
    """Rebuild the manager's ``ManagerPolicy`` from captured ``squad_picks`` rows.

    Returns ``(policy, None)`` or ``(None, reason)``.  ``is_starting`` is the
    authoritative flag; a row without it falls back to FPL's own convention that
    slots 1-11 start.
    """

    starters: list[tuple[int, int]] = []
    bench: list[tuple[int, int]] = []
    captain: list[int] = []
    vice: list[int] = []
    for row in rows:
        pid = int(row["player_id"])
        slot = int(row.get("position") or 0)
        is_starting = row.get("is_starting")
        if is_starting is None:
            is_starting = 1 <= slot <= ml.XI_SIZE
        (starters if is_starting else bench).append((slot, pid))
        if bool(row.get("is_captain")):
            captain.append(pid)
        if bool(row.get("is_vice_captain")):
            vice.append(pid)
    starters.sort()
    bench.sort()
    starter_ids = tuple(sorted(pid for _slot, pid in starters))
    bench_ids = [pid for _slot, pid in bench]
    if len(starter_ids) != ml.XI_SIZE:
        return None, f"the captured lineup has {len(starter_ids)} starters, expected {ml.XI_SIZE}"
    bench_gk = [pid for pid in bench_ids if str(positions.get(pid) or "") == "GKP"]
    bench_out = tuple(pid for pid in bench_ids if str(positions.get(pid) or "") != "GKP")
    if len(bench_gk) != 1:
        return None, f"the captured bench has {len(bench_gk)} goalkeepers, expected 1"
    if len(bench_out) != ml.OUTFIELD_BENCH_SIZE:
        return None, (
            f"the captured bench has {len(bench_out)} outfield players, "
            f"expected {ml.OUTFIELD_BENCH_SIZE}"
        )
    if len(captain) != 1:
        return None, f"the captured lineup names {len(captain)} captains, expected 1"
    if len(vice) != 1:
        return None, f"the captured lineup names {len(vice)} vice-captains, expected 1"
    return (
        ml.ManagerPolicy(
            starter_ids=starter_ids,
            bench_gk_id=int(bench_gk[0]),
            bench_outfield_order=bench_out,
            captain_id=int(captain[0]),
            vice_captain_id=int(vice[0]),
        ),
        None,
    )


def bench_boost_manager_state(
    conn: sqlite3.Connection,
    entry_id: int,
    planning_event: int,
    *,
    as_of: str | None = None,
) -> BenchBoostManagerState:
    """Source the manager facts from canonical accessors, read-only.

    The fifteen come from ``manager_worlds.resolve_squad`` (the accepted
    manager-world authority) and their POSITIONS AND CLUBS come from
    ``route_comparator.load_player_meta`` (the accepted player authority the
    route layer uses), so a position label and a club are canonical facts rather
    than caller assertions.  The lineup comes from the manager's captured
    ``squad_picks``: the planning event's own capture when it exists, otherwise
    the most recent earlier capture.  Anything the canonical state does not
    supply refuses -- the adapter never invents a squad, a lineup, a position,
    a club or a chip.
    """

    context = planning_module.get_planning_context(conn, int(entry_id), int(planning_event), as_of)
    squad_info = mw.resolve_squad(context, conn)
    squad_ids = tuple(sorted(int(pid) for pid in squad_info["squad_ids"]))
    resolved_positions = {
        int(pid): str(position)
        for pid, position in (squad_info["positions"] or {}).items()
        if position
    }
    chips = tuple(dict(row) for row in planning_module.chips_state(conn, int(entry_id), int(planning_event)))

    if len(squad_ids) != ml.SQUAD_SIZE or len(resolved_positions) != len(squad_ids):
        raise BenchBoostAdapterError(
            f"{BB_MANAGER_STATE_MISSING}: the canonical planning state supplies {len(squad_ids)} "
            f"players with {len(resolved_positions)} known positions, expected {ml.SQUAD_SIZE} of each "
            f"(squad_state={squad_info.get('squad_state')!r})",
            reasons=(BB_MANAGER_STATE_MISSING,),
        )

    positions, clubs = _canonical_positions_and_clubs(conn, squad_ids)
    # Two accepted accessors must agree about the same fifteen.  If they do not,
    # the player layer is not coherent and neither reading may be used.
    disagreements = sorted(
        pid for pid in squad_ids if str(resolved_positions.get(pid)) != str(positions.get(pid))
    )
    if disagreements:
        raise BenchBoostAdapterError(
            f"{BB_MANAGER_STATE_MISSING}: the squad and player authorities disagree about the position of "
            f"{disagreements[:8]}",
            reasons=(BB_MANAGER_STATE_MISSING,),
        )

    found = repo.latest_complete_squad_rows(conn, int(entry_id), int(planning_event))
    if found is None:
        raise BenchBoostAdapterError(
            f"{BB_LINEUP_NOT_CAPTURED}: no complete admissible lineup has been captured at or before "
            f"GW{int(planning_event)}; the XI, bench and armband are unknown and are never invented",
            reasons=(BB_LINEUP_NOT_CAPTURED,),
        )
    lineup_event, rows = found
    policy, reason = _lineup_policy(rows, positions)
    if policy is None:
        raise BenchBoostAdapterError(
            f"{BB_LINEUP_ILLEGAL}: the lineup captured for GW{int(lineup_event)} cannot be used: {reason}",
            reasons=(BB_LINEUP_ILLEGAL,),
        )
    if set(bb.squad_ids_of(policy)) != set(squad_ids):
        # The submitted lineup names a different fifteen than the canonical
        # squad, so it does not describe what the chip would actually affect.
        raise BenchBoostAdapterError(
            f"{BB_LINEUP_SQUAD_MISMATCH}: the lineup captured for GW{int(lineup_event)} names "
            f"{len(bb.squad_ids_of(policy))} players that are not the canonical squad of {len(squad_ids)}; "
            "a stale lineup is never remapped onto a changed squad",
            reasons=(BB_LINEUP_SQUAD_MISMATCH,),
        )

    state = BenchBoostManagerState(
        entry_id=int(entry_id),
        planning_event=int(planning_event),
        squad_ids=squad_ids,
        policy=policy,
        positions=positions,
        clubs=clubs,
        chip_availability=chips,
        lineup_source_event=int(lineup_event),
        lineup_is_exact_event=int(lineup_event) == int(planning_event),
    )
    problems = state.problems()
    if problems:
        raise BenchBoostAdapterError(
            f"{BB_MANAGER_STATE_MISSING}: canonical manager state is not usable: " + "; ".join(problems),
            reasons=(BB_MANAGER_STATE_MISSING,),
        )
    return state


# ---------------------------------------------------------------------------
# Certified predictive evidence (supplied, then validated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchBoostCertifiedInputs:
    """The certified predictive evidence a caller must supply.

    The adapter cannot MANUFACTURE these -- producing certified projections and
    worlds requires running the prediction pipeline, which this task must not do.
    It therefore accepts them and validates that the horizon binding is the
    canonical certified one for the planning event and that the world inputs are
    the SAME certified context, so a stale or cross-cutoff matrix cannot be
    evaluated as if it were current.
    """

    horizon_binding: cd.ChipHorizonBinding
    worlds: cd.ChipWorldInputs


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def build_bench_boost_request(
    manager: BenchBoostManagerState,
    certified: BenchBoostCertifiedInputs,
    *,
    conn: sqlite3.Connection | None = None,
    as_of: str | None = None,
    #: An explicit, loud opt-out for a caller that has NO database: a pure
    #: simulation or a unit test.  Without it (or a ``conn``) the request is
    #: refused, because positions, clubs and the lineup are canonical facts
    #: and a hand-made state cannot assert them.
    allow_unverified_manager_state: bool = False,
    calibration_status: str = cd.CALIBRATION_UNCALIBRATED,
    input_uncertainty_flags: Sequence[str] = (),
) -> bb.BenchBoostRequest:
    """Assemble an authoritative ``BenchBoostRequest``, or refuse.

    Nothing here is defaulted.  In particular:

      * the squad and lineup are the ones ``bench_boost_manager_state`` sourced
        from canonical state;
      * when ``conn`` is supplied the canonical state is RE-DERIVED and any
        disagreement refuses, so a caller cannot self-certify a manager squad
        while canonical manager authority exists;
      * the horizon binding must be the canonical certified four-event window
        for this planning event, and the world inputs must declare the same
        window, planning event and certification identity.
    """

    if conn is None and not allow_unverified_manager_state:
        raise BenchBoostAdapterError(
            f"{BB_CANONICAL_AUTHORITY_REQUIRED}: a request must be built against the canonical manager "
            "state; pass the connection, or set allow_unverified_manager_state=True to declare that "
            "this is a simulation with no database.  Caller-supplied positions, clubs and lineups are "
            "never authority over the canonical ones",
            reasons=(BB_CANONICAL_AUTHORITY_REQUIRED,),
        )

    problems = manager.problems()
    if problems:
        raise BenchBoostAdapterError(
            f"{BB_MANAGER_STATE_MISSING}: the supplied manager state is not usable: " + "; ".join(problems),
            reasons=(BB_MANAGER_STATE_MISSING,),
        )

    if conn is not None:
        canonical = bench_boost_manager_state(
            conn, int(manager.entry_id), int(manager.planning_event), as_of=as_of
        )
        disagreements: list[str] = []
        if tuple(canonical.squad_ids) != tuple(manager.squad_ids):
            disagreements.append(
                f"squad differs ({len(manager.squad_ids)} supplied vs {len(canonical.squad_ids)} canonical)"
            )
        if canonical.policy.ordering_key() != manager.policy.ordering_key():
            disagreements.append("the lineup/armband is not the captured canonical one")
        if int(canonical.lineup_source_event) != int(manager.lineup_source_event):
            disagreements.append(
                f"lineup source event {int(manager.lineup_source_event)} != canonical "
                f"{int(canonical.lineup_source_event)}"
            )
        # POSITIONAL and CLUB identity are part of the canonical squad, not a
        # caller's label map.  Relabelling one player's position would change
        # which autosubs are legal -- and therefore the chip's value -- while
        # keeping the same fifteen ids, and a club map is what makes the
        # three-per-club limit checkable at all.
        relabelled = sorted(
            pid
            for pid in manager.squad_ids
            if str(manager.positions.get(int(pid))) != str(canonical.positions.get(int(pid)))
        )
        if relabelled:
            disagreements.append(f"position of {relabelled[:8]} is not the canonical one")
        reclubbed = sorted(
            pid
            for pid in manager.squad_ids
            if int(manager.clubs.get(int(pid)) or 0) != int(canonical.clubs.get(int(pid)) or 0)
        )
        if reclubbed:
            disagreements.append(f"club of {reclubbed[:8]} is not the canonical one")
        if disagreements:
            raise BenchBoostAdapterError(
                f"{BB_CALLER_STATE_DISAGREES}: the supplied manager state is not the canonical one: "
                + "; ".join(disagreements),
                reasons=(BB_CALLER_STATE_DISAGREES,),
            )

    binding = certified.horizon_binding
    if int(binding.planning_event) != int(manager.planning_event):
        raise BenchBoostAdapterError(
            f"{BB_HORIZON_MISMATCH}: the certified horizon is bound to planning event "
            f"{int(binding.planning_event)} but the manager state is for GW{int(manager.planning_event)}",
            reasons=(BB_HORIZON_MISMATCH,),
        )
    if bool(manager.lineup_is_exact_event) is False and int(manager.lineup_source_event) != int(
        manager.planning_event
    ):
        # Not an error: before the coming deadline the manager's most recent
        # submitted lineup is the correct authority.  ``lineup_source_event``
        # carries that fact into the report; nothing is silently substituted.
        pass

    request = bb.BenchBoostRequest(
        worlds=certified.worlds,
        horizon_binding=binding,
        policy=manager.policy,
        positions=dict(manager.positions),
        chip_available=_chip_available(manager.chip_availability, planning_event=int(manager.planning_event)),
        calibration_status=str(calibration_status),
        input_uncertainty_flags=tuple(str(flag) for flag in input_uncertainty_flags),
    )
    # Validate eagerly: the caller learns about a non-canonical horizon or an
    # uncovered player here rather than deep inside the evaluation.
    cd.ChipHorizonBinding.validate(binding)
    problems = binding.matches_worlds(certified.worlds)
    if problems:
        raise BenchBoostAdapterError(
            f"{BB_HORIZON_MISMATCH}: the certified worlds are not the bound context: " + "; ".join(problems),
            reasons=(BB_HORIZON_MISMATCH,),
        )
    return request


def _chip_available(chip_availability: Sequence[Mapping[str, Any]], *, planning_event: int) -> bool:
    """Whether Bench Boost is genuinely eligible for this event.

    Delegates to the arbiter's own per-row predicate, so availability cannot
    drift from the eligibility that admits the action.  Only rows for the
    ``bboost`` definition are considered, and a chip with two seasonal windows
    is eligible when ANY of its rows is (a first-half use does not consume the
    second-half allocation).
    """

    rows = [row for row in chip_availability if str(row.get("name") or "") == cd.CHIP_ACTION_TO_OFFICIAL_NAME[cd.CHIP_ACTION_BB]]
    if not rows:
        return False
    mapped = cd._availability_by_action(rows, planning_event=int(planning_event))
    entry = mapped.get(cd.CHIP_ACTION_BB) or {}
    return bool(entry.get("eligible"))
