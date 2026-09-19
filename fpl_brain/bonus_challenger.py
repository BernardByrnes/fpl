"""PE-3 — the structural BONUS/BPS challenger (CHALLENGER ONLY).

Version: ``bonus_bps_v1.0.0``.  Grain: PLAYER x FIXTURE x WORLD.

This is a STRUCTURAL APPROXIMATION, never an exact BPS reconstruction.  Bonus is a fixture
competition, so the model RANKS every player in a simulated fixture together and then
applies the already-accepted canonical allocator; a player's bonus is never predicted
independently of everyone else in his match.

WHAT IS MODELLED STRUCTURALLY
    only the events the current simulator genuinely produces: minutes, goals, assists,
    clean sheets, goals conceded, saves, yellow cards.  A rule whose primitive the
    simulation does not produce is SKIPPED AND NAMED (``unsupported_rule_rows``) rather
    than counted as zero, and the challenger reports the limitation flags for them.

THE BACKGROUND / RESIDUAL EQUATION (the double-counting guard)
    Historical total BPS already CONTAINS goals, assists, clean sheets, saves and every
    other action.  Adding simulated event BPS on top of a historical total-BPS level would
    therefore count those actions twice.  The construction is CENTRED instead::

        world_bps_i  =  background_expected_bps_i
                       + SUM_c ( structural_bps_ic(world) - expected_structural_bps_ic )

    where ``expected_structural_bps_ic`` is the mean of component c for that player over
    the sampled worlds.  By construction the second term has expectation zero, so::

        E[world_bps_i] = background_expected_bps_i

    The historical level supplies the MEAN; the simulated events only move the player
    above or below it.  That identity is asserted in the tests, which is what proves no
    uncentred double count can occur.

Nothing here replaces the certified ``bonus_xpts`` path.  This module is a challenger and
is imported by nothing in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import bonus_allocation as ba
from . import bps_rules as bps

BONUS_BPS_MODEL_VERSION = "bonus_bps_v1.0.0"

#: The world events the current simulator ACTUALLY produces, per player per fixture.
#: Everything else is unsupported and reported as such.  MEASURED against monte_carlo.py:
#: minutes, goals (via the calibrated scorer allocation), assists, clean-sheet state,
#: goals conceded while on the pitch, raw save count, yellow cards.
STRUCTURAL_EVENT_NAMES: tuple[str, ...] = (
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
    "yellow_cards",
)

#: BPS components the challenger CANNOT model, with the reason.  Surfaced on every result;
#: the brief requires the unsupported set to be explicit rather than implied.
UNSUPPORTED_COMPONENT_FLAGS: tuple[tuple[str, str], ...] = (
    ("PENALTY_GOAL_BPS_UNRESOLVED",
     "the simulator does not distinguish a penalty goal from an open-play goal, so the "
     "direct-penalty-goal row cannot be applied; goals are scored at the positional "
     "non-penalty value and the difference is unmodelled"),
    ("SAVE_LOCATION_BPS_UNRESOLVED",
     "the simulator samples a save COUNT only; inside-box saves are not produced, so the "
     "+1 inside-box row is unsupported"),
    ("BIG_CHANCE_SAVE_BPS_UNRESOLVED",
     "big-chance saves are not produced by the simulator"),
)


def unsupported_component_names() -> tuple[str, ...]:
    return tuple(flag for flag, _reason in UNSUPPORTED_COMPONENT_FLAGS)


def unsupported_component_flags(rule_rows: Iterable[str]) -> tuple[str, ...]:
    """Map unsupported RULE ROWS onto the limitation flags a consumer can act on."""

    rows = set(rule_rows)
    flags = []
    if "direct_penalty_goal" in rows:
        flags.append("PENALTY_GOAL_BPS_UNRESOLVED")
    if "save_from_inside_box" in rows:
        flags.append("SAVE_LOCATION_BPS_UNRESOLVED")
    if "save_from_big_chance" in rows:
        flags.append("BIG_CHANCE_SAVE_BPS_UNRESOLVED")
    return tuple(flags)


@dataclass(frozen=True)
class PlayerWorldBps:
    """One player's structural BPS in one world, with the unsupported rows named."""

    player_id: int
    bps: float
    unsupported_rule_rows: tuple[str, ...]
    flags: tuple[str, ...] = ()


def structural_world_bps(player_id: int, position: str,
                         world_events: Mapping[str, Any],
                         rules: Sequence[bps.BPSPrimitiveSpec] = bps.RULE_SPECS,
                         ) -> PlayerWorldBps:
    """Structural BPS for one player in one simulated world.

    ``world_events`` must carry at least every name in :data:`STRUCTURAL_EVENT_NAMES`; a
    missing STRUCTURAL event is a programming error and raises, because the challenger is
    supposed to be fed the simulator's own outputs.
    """

    missing = [name for name in STRUCTURAL_EVENT_NAMES if name not in world_events]
    if missing:
        raise bps.BPSPrimitiveMissing(position, missing)
    points, skipped = bps.structural_bps(position, dict(world_events), rules)
    return PlayerWorldBps(player_id=int(player_id), bps=int(points),
                          unsupported_rule_rows=skipped,
                          flags=unsupported_component_flags(skipped))


# ---------------------------------------------------------------------------
# Background BPS level — causal, pre-cutoff only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackgroundBps:
    """A shrunk causal BPS-per-90 level, and the evidence it was built from."""

    bps_per_90: float
    sample_minutes: int
    source_rows: int
    shrinkage_minutes: int

    def expected_bps(self, expected_minutes: float) -> float:
        """The historical level a player is expected to sit at for those minutes."""

        return self.bps_per_90 * float(expected_minutes) / 90.0


def background_bps_from_rows(rows: Iterable[Mapping[str, Any]], *,
                             shrinkage_minutes: int = 900,
                             fallback_per_90: float = 0.0) -> BackgroundBps:
    """Build the background level from CAUSAL historical rows only.

    The caller is responsible for supplying rows that respect the frozen PE-1 causal
    boundary — scheduled placeholders and post-cutoff rows must already be excluded.  This
    function deliberately contains NO history predicate of its own, so it cannot invent a
    second, divergent one.

    Shrinkage is minutes-weighted toward ``fallback_per_90`` with a pseudo-count of
    ``shrinkage_minutes``, so a handful of minutes cannot dominate.
    """

    minutes = 0
    bps_total = 0
    count = 0
    for row in rows:
        row_minutes = int(row.get("minutes") or 0)
        if row_minutes <= 0:
            continue
        minutes += row_minutes
        bps_total += int(row.get("bps") or 0)
        count += 1
    if minutes <= 0:
        return BackgroundBps(bps_per_90=float(fallback_per_90), sample_minutes=0,
                             source_rows=0, shrinkage_minutes=int(shrinkage_minutes))
    raw = bps_total * 90.0 / minutes
    weight = minutes / (minutes + float(shrinkage_minutes))
    shrunk = weight * raw + (1.0 - weight) * float(fallback_per_90)
    return BackgroundBps(bps_per_90=shrunk, sample_minutes=minutes, source_rows=count,
                         shrinkage_minutes=int(shrinkage_minutes))


def centred_world_bps(background_expected_bps: float,
                      structural_by_world: Sequence[float]) -> list[float]:
    """The double-counting guard, as one function.

    ``world_bps_w = background + (structural_w - mean(structural))`` so the simulated
    events only DEVIATE a player from his historical level and never re-add it.  The mean
    of the returned series equals ``background_expected_bps`` exactly.
    """

    if not structural_by_world:
        return []
    mean_structural = sum(structural_by_world) / len(structural_by_world)
    return [float(background_expected_bps) + (value - mean_structural)
            for value in structural_by_world]


# ---------------------------------------------------------------------------
# World-level bonus — the fixture competition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureWorldBonus:
    """One world's joint allocation over every player in the fixture."""

    bps_by_player: Mapping[int, float]
    bonus_by_player: Mapping[int, int]
    total_bonus: int


def allocate_fixture_world(bps_by_player: Mapping[int, float]) -> FixtureWorldBonus:
    """Rank the WHOLE fixture and allocate — the canonical allocator, unchanged."""

    bonus = ba.allocate_fixture_bonus(bps_by_player)
    return FixtureWorldBonus(bps_by_player=dict(bps_by_player), bonus_by_player=bonus,
                             total_bonus=sum(bonus.values()))


@dataclass
class WorldBonusSummary:
    """Per-player aggregation over worlds, plus the fixture competition diagnostics."""

    expected_bonus: dict[int, float] = field(default_factory=dict)
    p_bonus_any: dict[int, float] = field(default_factory=dict)
    p_bonus_2plus: dict[int, float] = field(default_factory=dict)
    p_bonus_3: dict[int, float] = field(default_factory=dict)
    mean_bps_proxy: dict[int, float] = field(default_factory=dict)
    worlds: int = 0
    ranking_changed_across_worlds: bool = False
    unsupported_rule_rows: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    total_bonus_per_world: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": BONUS_BPS_MODEL_VERSION,
            "grain": "player_x_fixture_x_world",
            "worlds": self.worlds,
            "expected_bonus": self.expected_bonus,
            "p_bonus_any": self.p_bonus_any,
            "p_bonus_2plus": self.p_bonus_2plus,
            "p_bonus_3": self.p_bonus_3,
            "mean_bps_proxy": self.mean_bps_proxy,
            "ranking_changed_across_worlds": self.ranking_changed_across_worlds,
            "unsupported_rule_rows": list(self.unsupported_rule_rows),
            "flags": list(self.flags),
        }


def aggregate_worlds(per_world_bps: Sequence[Mapping[int, float]],
                     per_world_unsupported: Iterable[Iterable[str]] = (),
                     ) -> WorldBonusSummary:
    """Apply the fixture competition per world, then aggregate across worlds."""

    summary = WorldBonusSummary()
    if not per_world_bps:
        return summary

    players = sorted({int(pid) for world in per_world_bps for pid in world})
    bonus_totals = {pid: 0.0 for pid in players}
    bps_totals = {pid: 0.0 for pid in players}
    p_any = {pid: 0.0 for pid in players}
    p_two = {pid: 0.0 for pid in players}
    p_three = {pid: 0.0 for pid in players}
    top_orderings: set[tuple[int, ...]] = set()
    world_totals: list[int] = []
    skipped: set[str] = set()

    for world in per_world_bps:
        allocation = allocate_fixture_world(world)
        world_totals.append(allocation.total_bonus)
        for player_id in players:
            bps_value = float(world.get(player_id, 0.0))
            bonus = int(allocation.bonus_by_player.get(player_id, 0))
            bps_totals[player_id] += bps_value
            bonus_totals[player_id] += bonus
            if bonus > 0:
                p_any[player_id] += 1.0
            if bonus >= 2:
                p_two[player_id] += 1.0
            if bonus == 3:
                p_three[player_id] += 1.0
        ranked = sorted(players, key=lambda pid: -float(world.get(pid, 0.0)))[:3]
        top_orderings.add(tuple(ranked))

    for skipped_rows in per_world_unsupported:
        skipped.update(skipped_rows)

    worlds = len(per_world_bps)
    summary.worlds = worlds
    summary.expected_bonus = {pid: bonus_totals[pid] / worlds for pid in players}
    summary.mean_bps_proxy = {pid: bps_totals[pid] / worlds for pid in players}
    summary.p_bonus_any = {pid: p_any[pid] / worlds for pid in players}
    summary.p_bonus_2plus = {pid: p_two[pid] / worlds for pid in players}
    summary.p_bonus_3 = {pid: p_three[pid] / worlds for pid in players}
    summary.ranking_changed_across_worlds = len(top_orderings) > 1
    summary.unsupported_rule_rows = tuple(sorted(skipped))
    summary.flags = unsupported_component_flags(skipped)
    summary.total_bonus_per_world = tuple(world_totals)
    return summary
