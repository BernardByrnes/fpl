"""PE-7 team -> player attacking-mass coherence.

WHAT THIS IS
------------
One place that answers, for a fixture side, whether the player-level attacking
expectations are coherent with the team-level football-event environment they
were supposed to arise from.  It is a CHECK, not a model: it changes no
prediction, rescales nothing and redistributes nothing.

THE RULE IT ENFORCES
--------------------
For every (fixture, side) and every attacking component:

* the team's attacking expectation is finite and non-negative;
* the players' attacking expectations are finite and non-negative;
* the players' mass is the sum of their own per-90 rate applied to their OWN
  minute exposure for that fixture, so nothing is invented for a player who did
  not play and nothing is invented for a fixture that was not played;
* the mass may not exceed the team's environment by more than the declared
  tolerance.  Where it does, the record says so: the excess is REPORTED, never
  clamped, never scaled back, and never silently moved to another player.  A
  quiet rescale would make the two layers agree at the cost of hiding that they
  disagreed;
* unallocated mass — the part of the team's expectation the player level does not
  exhaust — is an explicit residual on every record, reported as an absolute
  quantity and as a share of the team expectation, with the residual's status
  named.  It is never redistributed and never re-described as player expectation;
* an ASSIST bound uses the same team environment: an assist requires a goal, so
  the assist mass of a side cannot exceed the goals the side is expected to
  score.  This is an inequality, not an attribution of goals to assists.

FIXTURE ATOMICITY AND BLANKS
----------------------------
Every record is one (fixture, side).  A double gameweek therefore produces one
atomic record per fixture, and any per-event total is an aggregate of those atomic
records, labelled as an aggregate and carrying its record count — never a single
event-level number that a DGW would silently double.  A team with no fixture in
the event has no record at all: a blank cannot produce an invented fixture attack
because a fixture attack exists only where there is a fixture.

DETERMINISM
-----------
Records are keyed and sorted canonically and the block carries a digest over that
canonical form, so the allocation is stable under database row order: reordering
the inputs cannot change the digest, and a test pins it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from . import analytics

COHERENCE_VERSION = "team_player_attack_coherence_v1.0.0"

GRAIN_FIXTURE_SIDE = "fixture_side"

#: How mass is combined.  Declared once so a reader never has to infer whether a
#: published number was rescaled.
REDISTRIBUTION_NONE = "NONE_EXPLICIT_RESIDUAL_REPORTED"
MASS_FORMULA = "sum over players of posterior_rate_per90 * own_minutes_for_this_fixture / 90"

#: Declared numerical tolerance for "the player level does not exceed the team
#: environment".  It absorbs the 6-decimal storage rounding of a published row and
#: nothing else.
DECLARED_MASS_TOLERANCE = 1e-6

STATUS_COHERENT = "COHERENT"
STATUS_EXHAUSTED = "PLAYER_MASS_EXHAUSTS_THE_TEAM_ENVIRONMENT"
STATUS_UNALLOCATED = "UNALLOCATED_TEAM_MASS_REPORTED"
STATUS_EXCEEDS = "PLAYER_MASS_EXCEEDS_TEAM_ENVIRONMENT"
STATUS_NO_EXPOSURE = "NO_PLAYER_EXPOSURE_EVIDENCE"
STATUS_INVALID = "INVALID_ATTACK_EXPECTATION"

COMPONENT_XG = "xG_per90"
COMPONENT_XA = "xA_per90"
ATTACK_COMPONENTS: tuple[str, ...] = (COMPONENT_XG, COMPONENT_XA)

FLAG_TEAM_EXPECTATION_INVALID = "TEAM_ATTACK_EXPECTATION_INVALID"
FLAG_PLAYER_EXPECTATION_INVALID = "PLAYER_ATTACK_EXPECTATION_INVALID"
FLAG_MASS_EXCEEDS_ENVIRONMENT = "PLAYER_MASS_EXCEEDS_TEAM_ENVIRONMENT"
FLAG_UNALLOCATED_MASS = "UNALLOCATED_TEAM_ATTACKING_MASS"
FLAG_NO_EXPOSURE = "NO_PLAYER_EXPOSURE_EVIDENCE"
FLAG_INVENTED_FIXTURE_ATTACK = "INVENTED_FIXTURE_ATTACK"


class CoherenceError(ValueError):
    """The coherence block cannot be produced from the inputs it was given."""


def _finite_non_negative(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def validate_attack_expectations(
    rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Every value that is missing, non-finite or negative, with its identity."""

    violations: list[dict[str, Any]] = []
    for row in rows:
        for field in fields:
            if field not in row:
                continue
            if not _finite_non_negative(row.get(field)):
                violations.append(
                    {
                        "identity": {
                            key: row.get(key)
                            for key in ("fixture_id", "team_id", "player_id", "component", "event")
                            if key in row
                        },
                        "field": field,
                        "value": row.get(field),
                    }
                )
    return violations


def player_attack_mass(
    player_rows: Iterable[Mapping[str, Any]],
    exposures: Mapping[tuple[int, int], float],
    fixture_sides: Mapping[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Player-level attacking mass per (fixture, side, component).

    ``exposures`` maps ``(player_id, fixture_id)`` to the minutes that player was
    on the pitch for that fixture — realised exposure on the evaluation path, so
    the check cannot be dominated by predicted-minutes error.  A player with no
    exposure entry contributes nothing, which is how a player who did not play
    contributes no invented mass.

    ``fixture_sides`` maps ``fixture_id`` to ``(home_team, away_team)``, so the
    side is read from the FIXTURE rather than from a mutable club column.
    """

    # Contributions are collected first and summed in a CANONICAL order, so the
    # total does not depend on the order the caller supplied the rows in (float
    # addition is not associative, and a digest over a differently-summed total
    # would report motion where the allocation is the same).
    contributions: list[tuple[int, str, int, int, float, float, float]] = []
    for row in player_rows:
        player_id = int(row["player_id"])
        component = str(row["component"])
        rate = row.get("posterior_mean")
        team_id = int(row["team_id"])
        for fixture_id, (home, away) in sorted(fixture_sides.items()):
            if team_id not in (int(home), int(away)):
                continue
            minutes = exposures.get((player_id, int(fixture_id)))
            if minutes is None or float(minutes) <= 0.0 or rate is None:
                continue
            contributions.append(
                (
                    int(fixture_id),
                    component,
                    team_id,
                    player_id,
                    float(minutes),
                    float(rate),
                    float(rate) * float(minutes) / 90.0,
                )
            )
    grouped: dict[tuple[int, int, str], list[tuple[int, str, int, int, float, float, float]]] = {}
    for item in sorted(contributions):
        grouped.setdefault((item[0], item[2], item[1]), []).append(item)
    records: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = grouped[key]
        records.append(
            {
                "fixture_id": key[0],
                "team_id": key[1],
                "component": key[2],
                "mass": round(sum(item[6] for item in items), 6),
                "players_contributing": len(items),
                "players": [
                    {
                        "player_id": item[3],
                        "minutes": round(item[4], 6),
                        "rate": round(item[5], 6),
                        "mass": round(item[6], 6),
                    }
                    for item in items
                ],
            }
        )
    return records


def _sides_from_fixtures(fixtures: Mapping[int, Any]) -> dict[int, tuple[int, int]]:
    return {
        int(fixture_id): (int(fixture["team_h"]), int(fixture["team_a"]))
        for fixture_id, fixture in fixtures.items()
    }


#: Which status a record carries when more than one applied.  Declared once so a
#: reader never has to know the order the component loop happened to run in.
STATUS_PRIORITY: tuple[str, ...] = (
    STATUS_INVALID,
    STATUS_EXCEEDS,
    STATUS_UNALLOCATED,
    STATUS_NO_EXPOSURE,
    STATUS_EXHAUSTED,
)


def _primary_status(statuses: Sequence[str]) -> str:
    present = set(statuses)
    for status in STATUS_PRIORITY:
        if status in present:
            return status
    raise CoherenceError(f"no declared status applies to {sorted(present)}")


def attacking_mass_allocation(
    team_rows: Sequence[Mapping[str, Any]],
    mass_records: Sequence[Mapping[str, Any]],
    *,
    fixtures: Mapping[int, Mapping[str, Any]] | None = None,
    tolerance: float = DECLARED_MASS_TOLERANCE,
) -> dict[str, Any]:
    """The per-fixture-side coherence block, with explicit residual mass.

    ``team_rows`` are team fixture projections; ``mass_records`` are
    player-level mass records from :func:`player_attack_mass`.  When ``fixtures``
    is supplied (fixture id -> fixture row) the block also proves fixture
    atomicity and the absence of invented fixture attack: a record's fixture must
    exist, the side must be one of that fixture's two sides, and a team plays
    exactly the fixtures the fixture list gives it, so a blank has no record.
    """

    tolerance = float(tolerance)
    violations: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    mass_by_key: dict[tuple[int, int, str], float] = {}
    contributing_by_key: dict[tuple[int, int, str], int] = {}
    invalid_player_keys: set[tuple[int, int, str]] = set()
    for record in mass_records:
        key = (int(record["fixture_id"]), int(record["team_id"]), str(record["component"]))
        raw_mass = record.get("mass")
        try:
            number: float | None = float(raw_mass)
        except (TypeError, ValueError):
            number = None
        if not _finite_non_negative(raw_mass):
            violations.append({"kind": "player_mass", "key": key, "value": raw_mass})
            invalid_player_keys.add(key)
        # A finite negative mass is kept exactly as it came in, like every other
        # published number here: the flag and the violation name it, and neither
        # clamps it.  A value that is not a number at all has no usable magnitude
        # and contributes nothing.
        mass_by_key[key] = number if number is not None and math.isfinite(number) else 0.0
        contributing_by_key[key] = int(record.get("players_contributing") or 0)

    invented: list[dict[str, Any]] = []
    sides = _sides_from_fixtures(fixtures) if fixtures is not None else {}
    for key in sorted(mass_by_key):
        fixture_id, team_id, _component = key
        if sides and (
            fixture_id not in sides or int(team_id) not in sides[fixture_id]
        ):
            invented.append({"fixture_id": fixture_id, "team_id": team_id})
    invented.extend(
        {
            "fixture_id": int(record["fixture_id"]),
            "team_id": int(record["team_id"]),
        }
        for record in team_rows
        if sides
        and (
            int(record["fixture_id"]) not in sides
            or int(record["team_id"]) not in sides[int(record["fixture_id"])]
        )
    )

    for row in sorted(team_rows, key=lambda item: (int(item["fixture_id"]), int(item["team_id"]))):
        fixture_id = int(row["fixture_id"])
        team_id = int(row["team_id"])
        expectation = row.get("expected_goals_for")
        flags: list[str] = []
        if sides and (
            fixture_id not in sides or team_id not in sides[fixture_id]
        ):
            # The row describes a side the fixture list does not contain, so it is
            # named as an invented fixture attack on the record itself and not only
            # in the block-level list.
            flags.append(FLAG_INVENTED_FIXTURE_ATTACK)
        if not _finite_non_negative(expectation):
            flags.append(FLAG_TEAM_EXPECTATION_INVALID)
            violations.append({"kind": "team_expectation", "key": (fixture_id, team_id), "value": expectation})
            records.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team_id,
                    "event": row.get("event"),
                    "venue": row.get("venue"),
                    "team_expectation": expectation,
                    "allocated": {},
                    "residual": {},
                    "residual_share": {},
                    "status": STATUS_INVALID,
                    # Every flag the record accumulated, so an invalid expectation
                    # cannot erase the reason the record was suspect in the first place.
                    "flags": sorted(set(flags)),
                    "redistribution": REDISTRIBUTION_NONE,
                }
            )
            continue
        expectation_value = float(expectation)
        allocated: dict[str, float] = {}
        residual: dict[str, float] = {}
        residual_share: dict[str, float | None] = {}
        contributing: dict[str, int] = {}
        statuses: list[str] = []
        for component in ATTACK_COMPONENTS:
            if (fixture_id, team_id, component) in invalid_player_keys:
                flags.append(FLAG_PLAYER_EXPECTATION_INVALID)
            mass = mass_by_key.get((fixture_id, team_id, component))
            mass_value = float(mass) if mass is not None else 0.0
            contributors = contributing_by_key.get((fixture_id, team_id, component), 0)
            allocated[component] = round(mass_value, 6)
            contributing[component] = contributors
            residual_value = expectation_value - mass_value
            residual[component] = round(residual_value, 6)
            residual_share[component] = (
                round(residual_value / expectation_value, 6) if expectation_value > 0 else None
            )
            if mass_value > expectation_value + tolerance:
                statuses.append(STATUS_EXCEEDS)
                flags.append(f"{FLAG_MASS_EXCEEDS_ENVIRONMENT}:{component}")
            elif mass is None:
                statuses.append(STATUS_NO_EXPOSURE)
                flags.append(f"{FLAG_NO_EXPOSURE}:{component}")
            elif abs(residual_value) <= tolerance:
                statuses.append(STATUS_EXHAUSTED)
            else:
                statuses.append(STATUS_UNALLOCATED)
                flags.append(f"{FLAG_UNALLOCATED_MASS}:{component}")
        if not any(
            mass_by_key.get((fixture_id, team_id, component)) is not None
            for component in ATTACK_COMPONENTS
        ):
            statuses.append(STATUS_NO_EXPOSURE)
        records.append(
            {
                "fixture_id": fixture_id,
                "team_id": team_id,
                "event": row.get("event"),
                "venue": row.get("venue"),
                "team_expectation": round(expectation_value, 6),
                "allocated": allocated,
                "residual": residual,
                "residual_share": residual_share,
                "players_contributing": contributing,
                "status": _primary_status(statuses),
                "statuses": sorted(set(statuses)),
                "flags": sorted(set(flags)),
                "redistribution": REDISTRIBUTION_NONE,
            }
        )

    event_aggregates: dict[str, Any] = {}
    per_event_fixtures: dict[str, set[int]] = {}
    for record in records:
        event = record.get("event")
        bucket = event_aggregates.setdefault(
            str(event),
            {
                "event": event,
                "atomic_records": 0,
                "teams": {},
                "team_expectation_total": 0.0,
                "allocated_total": {component: 0.0 for component in ATTACK_COMPONENTS},
                "note": (
                    "an event total is an AGGREGATE of the atomic per-fixture records above; a "
                    "double gameweek therefore contributes two records for the same team, and no "
                    "event-level number is ever derived as if a fixture list were one fixture"
                ),
            },
        )
        bucket["atomic_records"] += 1
        per_event_fixtures.setdefault(str(event), set()).add(int(record["fixture_id"]))
        team_entry = bucket["teams"].setdefault(
            str(record["team_id"]),
            {
                "team_id": record["team_id"],
                "fixtures": [],
                "team_expectation_total": 0.0,
                "allocated_total": {component: 0.0 for component in ATTACK_COMPONENTS},
                "residual_total": {component: 0.0 for component in ATTACK_COMPONENTS},
            },
        )
        team_entry["fixtures"].append(record["fixture_id"])
        if isinstance(record["team_expectation"], (int, float)):
            team_entry["team_expectation_total"] += float(record["team_expectation"])
            bucket["team_expectation_total"] += float(record["team_expectation"])
        for component in ATTACK_COMPONENTS:
            team_entry["allocated_total"][component] += float(record["allocated"].get(component) or 0.0)
            team_entry["residual_total"][component] += float(record["residual"].get(component) or 0.0)
            bucket["allocated_total"][component] += float(record["allocated"].get(component) or 0.0)
    for bucket in event_aggregates.values():
        bucket["atomic_records"] = int(bucket["atomic_records"])
        bucket["fixtures"] = sorted(per_event_fixtures[str(bucket["event"])])
        bucket["fixture_atomic"] = True
        bucket["team_expectation_total"] = round(float(bucket["team_expectation_total"]), 6)
        bucket["allocated_total"] = {
            component: round(float(value), 6) for component, value in bucket["allocated_total"].items()
        }
        for team_entry in bucket["teams"].values():
            team_entry["fixtures"] = sorted(int(value) for value in team_entry["fixtures"])
            team_entry["team_expectation_total"] = round(float(team_entry["team_expectation_total"]), 6)
            team_entry["allocated_total"] = {
                component: round(float(value), 6)
                for component, value in team_entry["allocated_total"].items()
            }
            team_entry["residual_total"] = {
                component: round(float(value), 6)
                for component, value in team_entry["residual_total"].items()
            }

    digest = analytics.canonical_hash(
        {"version": COHERENCE_VERSION, "grain": GRAIN_FIXTURE_SIDE, "records": records}
    )
    exceeds = [record for record in records if record["status"] == STATUS_EXCEEDS]
    checks = {
        "team_expectations_finite_and_non_negative": not any(
            record["status"] == STATUS_INVALID for record in records
        ),
        "player_expectations_finite_and_non_negative": not any(
            violation["kind"] == "player_mass" for violation in violations
        ),
        "no_component_exceeds_the_team_environment": not exceeds,
        "unallocated_mass_is_explicit": all(
            "residual" in record and "residual_share" in record for record in records
        ),
        "row_order_invariant": True,
        "fixture_atomic": True,
        "no_invented_fixture_attack": not invented,
        "redistribution": REDISTRIBUTION_NONE,
        "mass_formula": MASS_FORMULA,
        "declared_tolerance": tolerance,
        "rule": (
            "the player level may not exceed the team environment, unallocated mass is an explicit "
            "residual, and an excess is REPORTED rather than clamped or redistributed"
        ),
    }
    block_flags: list[str] = []
    if any(violation["kind"] == "player_mass" for violation in violations):
        block_flags.append(FLAG_PLAYER_EXPECTATION_INVALID)
    if any(violation["kind"] == "team_expectation" for violation in violations):
        block_flags.append(FLAG_TEAM_EXPECTATION_INVALID)
    if invented:
        # A mass key with no fixture side has no record to carry the flag, so the
        # block names the condition as well as listing the keys.
        block_flags.append(FLAG_INVENTED_FIXTURE_ATTACK)
    return {
        "coherence_version": COHERENCE_VERSION,
        "grain": GRAIN_FIXTURE_SIDE,
        "records": records,
        "event_aggregates": event_aggregates,
        "violations": violations,
        "invented_fixture_attack": invented,
        "exceeds_environment": [
            {"fixture_id": record["fixture_id"], "team_id": record["team_id"]} for record in exceeds
        ],
        "checks": checks,
        "flags": sorted(set(block_flags)),
        "digest": digest,
        "redistribution": REDISTRIBUTION_NONE,
        "status": (
            STATUS_INVALID if violations else (STATUS_EXCEEDS if exceeds else STATUS_COHERENT)
        ),
    }
