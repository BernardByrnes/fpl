"""FPL Core process facts and the point-in-time feature lab.

LAYERING
--------
``OFFICIAL_FPL`` facts stay authoritative and are never overwritten.  The
external process data lands in its OWN fact layers::

    PlayerMatchProcess   one player-fixture row of process counts
    ShotFact             one shot
    TeamMatchProcess     one team-fixture row (and its opponent-facing view)

Every external number keeps its provider field names and schema provenance so a
reviewer can trace it back to the cached bytes.

PLACEHOLDERS ARE NOT OBSERVATIONS
---------------------------------
``playermatchstats.csv`` contains rows for players who did not play, with
``minutes_played = 0`` and every process field empty.  ``played`` is strictly
``minutes_played > 0``, and features are only built from played rows, so a
0-minute placeholder can never enter a rate.

DEFCON
------
Official FPL ``defensive_contribution`` remains the authoritative outcome and the
thresholds come from Brain's own scoring rules (DEF 10, MID/FWD 12, 2 points,
never stacking, action sets CBIT / CBIRT).  The external defensive actions are
PREDICTORS of that outcome, never a replacement definition of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .scoring_rules import DEFAULT_SCORING_RULES, ScoringRules

CORE_PROCESS_FEATURES_VERSION = "core_process_features_v1.0.0"

HIGH_QUALITY_CHANCE_PROXY_VERSION = "hqc_proxy_v1.0.0"
#: A BRAIN-DERIVED proxy, deliberately not named "big chance": the provider's
#: ``big_chances`` is a team-level field and no per-player big-chance field exists.
HIGH_QUALITY_CHANCE_XG_THRESHOLD = 0.20

#: Poisson action-count assumption behind P(threshold).  Labelled explicitly
#: because it is an assumption, not a calibrated result.
DEFCON_COUNT_MODEL = "poisson_per_match_v1"

ATTACKING_FEATURE_NAMES = (
    "shots_per90",
    "shots_on_target_per90",
    "chances_created_per90",
    "box_touches_per90",
    "xg_external_per90",
    "xa_external_per90",
    "xg_per_shot",
    "xgot_per_shot",
    "shot_accuracy",
    "shot_location_quality",
    "open_play_shot_share",
    "set_piece_shot_share",
    "finishing_minus_xg",
    "finishing_minus_xgot",
    "attacking_process_consistency",
    "recent_process_trend",
    "share_team_xg",
    "share_team_shots",
    "share_team_sot",
    "share_team_box_touches",
    "share_team_chances_created",
)

DEFCON_FEATURE_NAMES = (
    "defcon_per90",
    "tackles_per90",
    "interceptions_per90",
    "recoveries_per90",
    "blocks_per90",
    "clearances_per90",
    "duels_won_per90",
    "aerial_duels_won_per90",
    "defensive_action_rate",
    "opponent_possession",
    "opponent_attacking_pressure",
)

MINUTES_SIGNAL_NAMES = ("recent_full_match_rate", "mean_start_min", "mean_finish_min")


def _f(value: Any) -> float | None:
    """Parse a provider numeric, returning None for empty/absent values."""

    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _i(value: Any) -> int | None:
    parsed = _f(value)
    return None if parsed is None else int(round(parsed))


def _zero_if_none(value: float | None) -> float:
    return 0.0 if value is None else float(value)


# ---------------------------------------------------------------------------
# player-fixture process facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerMatchProcess:
    """One player-fixture process row.  ``played`` excludes 0-minute placeholders."""

    external_player_id: int
    match_id: str
    official_player_id: int | None
    official_fixture_id: int | None
    minutes_played: int
    start_min: int | None
    finish_min: int | None
    total_shots: float | None
    shots_on_target: float | None
    xg_external: float | None
    xa_external: float | None
    chances_created: float | None
    touches_opposition_box: float | None
    successful_dribbles: float | None
    big_chances_missed: float | None
    touches: float | None
    final_third_passes: float | None
    tackles_won: float | None
    interceptions: float | None
    recoveries: float | None
    blocks: float | None
    clearances: float | None
    headed_clearances: float | None
    dribbled_past: float | None
    duels_won: float | None
    duels_lost: float | None
    ground_duels_won: float | None
    aerial_duels_won: float | None
    was_fouled: float | None
    fouls_committed: float | None
    dispossessed: float | None
    defensive_contributions_external: float | None
    xgot: float | None
    distance_covered: float | None
    number_of_sprints: float | None
    top_speed: float | None
    provider_schema: str = CORE_PROCESS_FEATURES_VERSION
    source_class: str = "EXTERNAL_FPL_CORE_INSIGHTS"

    @property
    def played(self) -> bool:
        return int(self.minutes_played) > 0

    @property
    def started(self) -> bool:
        return self.played and self.start_min is not None and int(self.start_min) <= 1

    def external_defensive_action_sum(self, position: str, rules: ScoringRules | None = None) -> float | None:
        """The provider's count of the position's DEFCON action set.

        CBIT for defenders, CBIRT for midfielders/forwards, mirroring Brain's own
        action-set definition.  The provider field is ``tackles_won``, which is a
        slightly narrower notion than the official "tackles" the real DEFCON uses,
        so this is a PREDICTOR and is compared against the official target rather
        than substituted for it.
        """

        rules = rules or DEFAULT_SCORING_RULES
        action_set = rules.defcon_action_set_for(str(position).upper()) or ""
        if not self.played or not action_set:
            return None
        components = {
            "C": self.clearances,
            "B": self.blocks,
            "I": self.interceptions,
            "R": self.recoveries,
            "T": self.tackles_won,
        }
        total = 0.0
        for letter in action_set:
            value = components.get(letter)
            if value is None:
                return None
            total += float(value)
        return total

    def as_dict(self) -> dict[str, Any]:
        return {
            "external_player_id": int(self.external_player_id),
            "official_player_id": self.official_player_id,
            "match_id": self.match_id,
            "official_fixture_id": self.official_fixture_id,
            "minutes_played": int(self.minutes_played),
            "played": bool(self.played),
            "started": bool(self.started),
            "total_shots": self.total_shots,
            "shots_on_target": self.shots_on_target,
            "xg_external": self.xg_external,
            "xa_external": self.xa_external,
            "chances_created": self.chances_created,
            "touches_opposition_box": self.touches_opposition_box,
            "tackles_won": self.tackles_won,
            "interceptions": self.interceptions,
            "recoveries": self.recoveries,
            "blocks": self.blocks,
            "clearances": self.clearances,
            "defensive_contributions_external": self.defensive_contributions_external,
        }


def build_player_match_process(
    rows: Sequence[Mapping[str, Any]],
    *,
    player_crosswalk: Any,
    fixture_crosswalk: Any,
) -> tuple[PlayerMatchProcess, ...]:
    """Normalise provider rows, resolving identity through the crosswalks.

    A row whose fixture or player cannot be resolved deterministically is returned
    with ``None`` identity (and therefore cannot be used destructively); it is
    never guessed.
    """

    fixtures = {e.external_match_id: int(e.official_fixture_id) for e in fixture_crosswalk.entries}
    out: list[PlayerMatchProcess] = []
    for row in rows:
        external_id = _i(row.get("player_id"))
        if external_id is None:
            continue
        match_id = str(row.get("match_id") or "")
        official_player = None
        if player_crosswalk is not None:
            official_player = player_crosswalk.official_id_for(int(external_id))
        out.append(
            PlayerMatchProcess(
                external_player_id=int(external_id),
                match_id=match_id,
                official_player_id=None if official_player is None else int(official_player),
                official_fixture_id=fixtures.get(match_id),
                minutes_played=int(_zero_if_none(_i(row.get("minutes_played")))),
                start_min=_i(row.get("start_min")),
                finish_min=_i(row.get("finish_min")),
                total_shots=_f(row.get("total_shots")),
                shots_on_target=_f(row.get("shots_on_target")),
                xg_external=_f(row.get("xg")),
                xa_external=_f(row.get("xa")),
                chances_created=_f(row.get("chances_created")),
                touches_opposition_box=_f(row.get("touches_opposition_box")),
                successful_dribbles=_f(row.get("successful_dribbles")),
                big_chances_missed=_f(row.get("big_chances_missed")),
                touches=_f(row.get("touches")),
                final_third_passes=_f(row.get("final_third_passes")),
                tackles_won=_f(row.get("tackles_won")),
                interceptions=_f(row.get("interceptions")),
                recoveries=_f(row.get("recoveries")),
                blocks=_f(row.get("blocks")),
                clearances=_f(row.get("clearances")),
                headed_clearances=_f(row.get("headed_clearances")),
                dribbled_past=_f(row.get("dribbled_past")),
                duels_won=_f(row.get("duels_won")),
                duels_lost=_f(row.get("duels_lost")),
                ground_duels_won=_f(row.get("ground_duels_won")),
                aerial_duels_won=_f(row.get("aerial_duels_won")),
                was_fouled=_f(row.get("was_fouled")),
                fouls_committed=_f(row.get("fouls_committed")),
                dispossessed=_f(row.get("dispossessed")),
                defensive_contributions_external=_f(row.get("defensive_contributions")),
                xgot=_f(row.get("xgot")),
                distance_covered=_f(row.get("distance_covered")),
                number_of_sprints=_f(row.get("number_of_sprints")),
                top_speed=_f(row.get("top_speed")),
            )
        )
    return tuple(sorted(out, key=lambda r: (r.match_id, r.external_player_id)))


# ---------------------------------------------------------------------------
# shot facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotFact:
    match_id: str
    official_player_id: int | None
    official_fixture_id: int | None
    shot_index: int
    minute: int
    is_home: bool
    outcome: str
    situation: str
    body_part: str
    xg: float | None
    xgot: float | None
    start_x: float | None
    start_y: float | None
    goal_mouth_y: float | None
    goal_mouth_z: float | None
    goal_mouth_location: str


def build_shots(
    rows: Sequence[Mapping[str, Any]], *, player_crosswalk: Any, fixture_crosswalk: Any
) -> tuple[ShotFact, ...]:
    fixtures = {e.external_match_id: int(e.official_fixture_id) for e in fixture_crosswalk.entries}
    out: list[ShotFact] = []
    for row in rows:
        external_id = _i(row.get("player_id"))
        official_player = None
        if player_crosswalk is not None and external_id is not None:
            official_player = player_crosswalk.official_id_for(int(external_id))
        match_id = str(row.get("match_id") or "")
        out.append(
            ShotFact(
                match_id=match_id,
                official_player_id=None if official_player is None else int(official_player),
                official_fixture_id=fixtures.get(match_id),
                shot_index=int(_zero_if_none(_i(row.get("shot_index")))),
                minute=int(_zero_if_none(_i(row.get("minute")))),
                is_home=str(row.get("is_home")).strip().lower() == "true",
                outcome=str(row.get("outcome") or ""),
                situation=str(row.get("situation") or ""),
                body_part=str(row.get("body_part") or ""),
                xg=_f(row.get("xg")),
                xgot=_f(row.get("xgot")),
                start_x=_f(row.get("start_x")),
                start_y=_f(row.get("start_y")),
                goal_mouth_y=_f(row.get("goal_mouth_y")),
                goal_mouth_z=_f(row.get("goal_mouth_z")),
                goal_mouth_location=str(row.get("goal_mouth_location") or ""),
            )
        )
    return tuple(sorted(out, key=lambda s: (s.match_id, s.shot_index, s.minute)))


def high_quality_chance_proxy(
    shots: Sequence[ShotFact], *, threshold: float = HIGH_QUALITY_CHANCE_XG_THRESHOLD
) -> int:
    """BRAIN_DERIVED count of shots at or above an xG threshold.

    Named explicitly as a proxy and versioned: it is NOT the provider's
    ``big_chances`` field (which is team-level) and must not be presented as one.
    """

    return sum(1 for shot in shots if shot.xg is not None and float(shot.xg) >= float(threshold))


# ---------------------------------------------------------------------------
# team / opponent process
# ---------------------------------------------------------------------------

_TEAM_FIELDS = (
    "possession", "xg", "xg_open_play", "xg_set_play", "non_penalty_xg", "xgot",
    "total_shots", "shots_on_target", "big_chances", "big_chances_missed",
    "blocked_shots", "shots_inside_box", "shots_outside_box", "touches_in_opposition_box",
    "tackles_won", "interceptions", "blocks", "clearances", "duels_won",
    "successful_dribbles", "accurate_passes", "keeper_saves",
)


@dataclass(frozen=True)
class TeamMatchProcess:
    """One team's process row for one fixture, plus its opponent-facing view."""

    official_fixture_id: int
    event: int
    match_id: str
    team_id: int
    opponent_id: int
    is_home: bool
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    provider_schema: str = CORE_PROCESS_FEATURES_VERSION
    source_class: str = "EXTERNAL_FPL_CORE_INSIGHTS"

    def value(self, name: str) -> float | None:
        return self.metrics.get(name)

    def opponent_facing(self) -> dict[str, float | None]:
        """What this team's opponent was ALLOWED, named consistently.

        Derived, not provider-supplied: these are the opponent's view of this
        team's process output, i.e. what this team allowed.
        """

        return {
            "opponent_shots_allowed": self.value("total_shots"),
            "opponent_sot_allowed": self.value("shots_on_target"),
            "opponent_box_shots_allowed": self.value("shots_inside_box"),
            "opponent_box_touches_allowed": self.value("touches_in_opposition_box"),
            "opponent_npxg_allowed": self.value("non_penalty_xg"),
            "opponent_xg_allowed": self.value("xg"),
            "opponent_big_chances_allowed": self.value("big_chances"),
            "opponent_possession_environment": self.value("possession"),
        }


def build_team_match_process(
    rows: Sequence[Mapping[str, Any]], *, fixture_crosswalk: Any, official_teams: Sequence[Mapping[str, Any]]
) -> tuple[TeamMatchProcess, ...]:
    """One row per (fixture, team) from the match-level provider block."""

    teams_by_code = {int(t["code"]): int(t["id"]) for t in official_teams if t.get("code") is not None}
    by_match = {e.external_match_id: e for e in fixture_crosswalk.entries}
    out: list[TeamMatchProcess] = []
    for row in rows:
        match_id = str(row.get("match_id") or "")
        entry = by_match.get(match_id)
        if entry is None:
            continue
        for side, team_id, opponent_id in (
            ("home", entry.home_team_id, entry.away_team_id),
            ("away", entry.away_team_id, entry.home_team_id),
        ):
            metrics = {name: _f(row.get(f"{side}_{name}")) for name in _TEAM_FIELDS}
            out.append(
                TeamMatchProcess(
                    official_fixture_id=int(entry.official_fixture_id),
                    event=int(entry.event),
                    match_id=match_id,
                    team_id=int(team_id),
                    opponent_id=int(opponent_id),
                    is_home=(side == "home"),
                    metrics=metrics,
                )
            )
    return tuple(sorted(out, key=lambda t: (t.event, t.official_fixture_id, t.team_id)))


# ---------------------------------------------------------------------------
# shrinkage
# ---------------------------------------------------------------------------


def shrunken_rate(
    *, total: float, exposure: float, prior_rate: float, prior_exposure: float
) -> float:
    """Minutes-equivalent empirical-Bayes shrinkage.

    ``(total + prior_rate * prior_exposure) / (exposure + prior_exposure)`` — the
    same idiom the existing player-rate model uses, so a tiny sample cannot
    produce a large rate.
    """

    exposure = float(exposure)
    prior_exposure = float(prior_exposure)
    if exposure <= 0.0 and prior_exposure <= 0.0:
        return float(prior_rate)
    return (float(total) + float(prior_rate) * prior_exposure) / (exposure + prior_exposure)


def shrunken_per90(
    *, total: float, minutes: float, prior_per90: float, prior_minutes: float
) -> float:
    """Minutes-equivalent shrinkage expressed PER 90 MINUTES.

    shrunken_rate returns a rate per unit of exposure, which for minutes is a
    per-minute rate; a feature named *_per90 must be scaled, so the conversion
    lives here rather than at each call site.
    """

    return shrunken_rate(
        total=total, exposure=minutes, prior_rate=float(prior_per90) / 90.0, prior_exposure=prior_minutes
    ) * 90.0


def _per90(total: float, minutes: float) -> float | None:
    if minutes <= 0.0:
        return None
    return float(total) / float(minutes) * 90.0


def _share(part: float | None, whole: float | None) -> float | None:
    if part is None or whole is None or float(whole) <= 0.0:
        return None
    return float(part) / float(whole)


# ---------------------------------------------------------------------------
# attacking feature lab
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackingFeatures:
    official_player_id: int
    matches_used: int
    minutes: float
    features: Mapping[str, float | None]
    coverage: Mapping[str, int]
    version: str = CORE_PROCESS_FEATURES_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "official_player_id": int(self.official_player_id),
            "matches_used": int(self.matches_used),
            "minutes": float(self.minutes),
            "features": dict(self.features),
            "coverage": dict(self.coverage),
            "version": self.version,
        }


def attacking_features(
    rows: Sequence[PlayerMatchProcess],
    *,
    team_process_by_fixture: Mapping[int, TeamMatchProcess] | None = None,
    prior_exposure: float = 900.0,
    recent_window: int = 2,
) -> AttackingFeatures:
    """Point-in-time attacking process features for one player.

    ``rows`` must already be restricted to what was available at the cutoff — this
    function never decides availability.  Placeholders are excluded, and every
    rate is shrunk toward a small-sample prior so a three-minute cameo cannot
    dominate.
    """

    if not rows:
        raise ValueError("attacking_features requires at least one player-match row")
    player_id = int(rows[0].official_player_id or -1)
    played = [row for row in sorted(rows, key=lambda r: (r.official_fixture_id or 0, r.match_id)) if row.played]
    minutes = float(sum(row.minutes_played for row in played))
    features: dict[str, float | None] = {name: None for name in ATTACKING_FEATURE_NAMES}
    coverage = {name: 0 for name in ATTACKING_FEATURE_NAMES}
    if not played or minutes <= 0.0:
        return AttackingFeatures(official_player_id=player_id, matches_used=len(played), minutes=minutes,
                                 features=features, coverage=coverage)

    def total(attr: str) -> tuple[float, int]:
        values = [getattr(row, attr) for row in played]
        present = [float(v) for v in values if v is not None]
        return sum(present), len(present)

    shots, n_shots = total("total_shots")
    sot, n_sot = total("shots_on_target")
    chances, n_chances = total("chances_created")
    box_touches, n_box = total("touches_opposition_box")
    xg, n_xg = total("xg_external")
    xa, n_xa = total("xa_external")
    xgot, n_xgot = total("xgot")

    # Per-90 rates, shrunk toward a zero-mean small-sample prior (prior_exposure
    # minutes of "no events"), so early-season rates stay conservative.
    for name, (value, count) in (
        ("shots_per90", (shots, n_shots)),
        ("shots_on_target_per90", (sot, n_sot)),
        ("chances_created_per90", (chances, n_chances)),
        ("box_touches_per90", (box_touches, n_box)),
        ("xg_external_per90", (xg, n_xg)),
        ("xa_external_per90", (xa, n_xa)),
    ):
        if count:
            features[name] = shrunken_per90(total=value, minutes=minutes, prior_per90=0.0,
                                            prior_minutes=prior_exposure)
            coverage[name] = count

    if n_shots:
        features["xg_per_shot"] = shrunken_rate(total=xg, exposure=shots, prior_rate=0.0,
                                                prior_exposure=1.0)
        features["shot_accuracy"] = shrunken_rate(total=sot, exposure=shots, prior_rate=0.0,
                                                  prior_exposure=1.0)
        coverage["xg_per_shot"] = coverage["shot_accuracy"] = n_shots
    if n_xgot:
        features["xgot_per_shot"] = shrunken_rate(total=xgot, exposure=shots, prior_rate=0.0,
                                                  prior_exposure=1.0)
        coverage["xgot_per_shot"] = n_xgot

    # finishing versus expected needs an OFFICIAL goals figure, which the caller
    # supplies from official FPL data; without it the feature stays None rather
    # than being inferred from the provider.
    return AttackingFeatures(
        official_player_id=player_id,
        matches_used=len(played),
        minutes=minutes,
        features=features,
        coverage=coverage,
    )


def attacking_features_with_shots(
    rows: Sequence[PlayerMatchProcess],
    shots: Sequence[ShotFact],
    *,
    team_process_by_fixture: Mapping[int, TeamMatchProcess] | None = None,
    official_goals: float | None = None,
    prior_exposure: float = 900.0,
    recent_window: int = 2,
) -> AttackingFeatures:
    """Attacking features plus the shot-derived and share-of-team families.

    official_goals is the OFFICIAL FPL goal count over the same window; it is
    required for the finishing-versus-expected family, which is left None when
    it is not supplied rather than being taken from the provider.
    """

    base = attacking_features(rows, team_process_by_fixture=team_process_by_fixture,
                              prior_exposure=prior_exposure, recent_window=recent_window)
    features = dict(base.features)
    coverage = dict(base.coverage)
    player_shots = [shot for shot in shots if base.official_player_id >= 0
                    and shot.official_player_id == base.official_player_id]
    if player_shots:
        xs = [float(s.start_x) for s in player_shots if s.start_x is not None]
        if xs:
            features["shot_location_quality"] = sum(xs) / len(xs)
            coverage["shot_location_quality"] = len(xs)
        situations = [s.situation.lower() for s in player_shots if s.situation]
        if situations:
            features["open_play_shot_share"] = sum(1 for s in situations if "regular" in s or "open" in s) / len(situations)
            features["set_piece_shot_share"] = sum(1 for s in situations if "set" in s or "corner" in s or "free" in s) / len(situations)
            coverage["open_play_shot_share"] = coverage["set_piece_shot_share"] = len(situations)
        hqc = high_quality_chance_proxy(player_shots)
        features["high_quality_chance_proxy"] = float(hqc)
        coverage["high_quality_chance_proxy"] = len(player_shots)

    # --- share of the team's process, over the fixtures he actually played -----
    if team_process_by_fixture:
        played_fixtures = [row for row in rows if row.played and row.official_fixture_id is not None]
        team = [team_process_by_fixture.get(int(row.official_fixture_id)) for row in played_fixtures]
        team = [row for row in team if row is not None]
        if team:
            def team_total(metric: str) -> float:
                return sum(float(t.value(metric) or 0.0) for t in team)

            # Player and team are both expressed per team-fixture over the SAME
            # fixtures, so games missed do not distort the share.
            for name, player_feature, team_metric in (
                ("share_team_xg", "xg_external_per90", "xg"),
                ("share_team_shots", "shots_per90", "total_shots"),
                ("share_team_sot", "shots_on_target_per90", "shots_on_target"),
                ("share_team_box_touches", "box_touches_per90", "touches_in_opposition_box"),
                ("share_team_chances_created", "chances_created_per90", "chances_created"),
            ):
                player_rate = base.features.get(player_feature)
                denominator = team_total(team_metric)
                if player_rate is None or denominator <= 0.0:
                    continue
                team_rate = denominator / float(len(team))
                if team_rate <= 0.0:
                    continue
                features[name] = float(player_rate) / float(team_rate)
                coverage[name] = len(team)

    if official_goals is not None and base.minutes > 0:
        goals_per90 = float(official_goals) / base.minutes * 90.0
        if base.features.get("xg_external_per90") is not None:
            features["finishing_minus_xg"] = goals_per90 - float(base.features["xg_external_per90"])
            coverage["finishing_minus_xg"] = coverage.get("xg_external_per90", 0)
        if base.features.get("xgot_per_shot") is not None:
            features["finishing_minus_xgot"] = goals_per90 - float(base.features["xgot_per_shot"])
            coverage["finishing_minus_xgot"] = coverage.get("xgot_per_shot", 0)

    # consistency: how evenly shots are spread across appearances
    played = [row for row in rows if row.played]
    if len(played) >= 2:
        counts = [float(row.total_shots or 0.0) for row in played]
        mean = sum(counts) / len(counts)
        if mean > 0:
            var = sum((c - mean) ** 2 for c in counts) / len(counts)
            features["attacking_process_consistency"] = 1.0 / (1.0 + math.sqrt(var))
            coverage["attacking_process_consistency"] = len(played)
        recent = counts[-int(recent_window):]
        features["recent_process_trend"] = (sum(recent) / len(recent)) - mean
        coverage["recent_process_trend"] = len(played)

    return AttackingFeatures(
        official_player_id=base.official_player_id,
        matches_used=base.matches_used,
        minutes=base.minutes,
        features=features,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# DEFCON feature lab
# ---------------------------------------------------------------------------


def defcon_threshold(position: str, rules: ScoringRules | None = None) -> int | None:
    """The official per-position DEFCON threshold, from Brain's scoring rules."""

    rules = rules or DEFAULT_SCORING_RULES
    return rules.defcon_threshold_for(str(position).upper())


@dataclass(frozen=True)
class DefconFeatures:
    """DEFCON predictors plus the derived posterior/threshold quantities."""

    official_player_id: int
    position: str
    threshold: int | None
    posterior_action_rate: float | None
    projected_defcon: float | None
    p_defcon_threshold: float | None
    expected_defcon_points: float | None
    defcon_variance: float | None
    predictors: Mapping[str, float | None]
    minutes: float
    matches_used: int
    count_model: str = DEFCON_COUNT_MODEL
    version: str = CORE_PROCESS_FEATURES_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "official_player_id": int(self.official_player_id),
            "position": self.position,
            "threshold": self.threshold,
            "posterior_action_rate": self.posterior_action_rate,
            "projected_defcon": self.projected_defcon,
            "p_defcon_threshold": self.p_defcon_threshold,
            "expected_defcon_points": self.expected_defcon_points,
            "defcon_variance": self.defcon_variance,
            "predictors": dict(self.predictors),
            "minutes": float(self.minutes),
            "matches_used": int(self.matches_used),
            "count_model": self.count_model,
            "version": self.version,
        }


def poisson_tail(mean: float, threshold: int) -> float:
    """P(X >= threshold) for X ~ Poisson(mean) — the explicit count assumption."""

    if threshold <= 0:
        return 1.0
    mean = max(0.0, float(mean))
    cumulative = 0.0
    term = math.exp(-mean)
    for k in range(threshold):
        if k > 0:
            term *= mean / k
        cumulative += term
    return max(0.0, min(1.0, 1.0 - cumulative))


def defcon_features(
    rows: Sequence[PlayerMatchProcess],
    *,
    position: str,
    expected_minutes: float,
    rules: ScoringRules | None = None,
    prior_action_rate: float = 0.0,
    prior_exposure: float = 900.0,
    opponent: TeamMatchProcess | None = None,
) -> DefconFeatures:
    """Predict the official DEFCON outcome from external defensive actions.

    The threshold and points come from Brain's scoring rules; the action set is
    the position's CBIT/CBIRT set; shrinkage is aggressive because early-season
    exposure is tiny.  The provider's ``defensive_contributions`` column is also
    carried as a predictor, but the authoritative outcome remains official FPL.
    """

    rules = rules or DEFAULT_SCORING_RULES
    threshold = defcon_threshold(position, rules)
    played = [row for row in rows if row.played]
    minutes = float(sum(row.minutes_played for row in played))
    predictors: dict[str, float | None] = {name: None for name in DEFCON_FEATURE_NAMES}

    def summed(attr: str) -> tuple[float, int]:
        values = [getattr(row, attr) for row in played]
        present = [float(v) for v in values if v is not None]
        return sum(present), len(present)

    for name, attr in (
        ("tackles_per90", "tackles_won"),
        ("interceptions_per90", "interceptions"),
        ("recoveries_per90", "recoveries"),
        ("blocks_per90", "blocks"),
        ("clearances_per90", "clearances"),
        ("duels_won_per90", "duels_won"),
        ("aerial_duels_won_per90", "aerial_duels_won"),
    ):
        total, count = summed(attr)
        if count and minutes > 0:
            predictors[name] = shrunken_per90(total=total, minutes=minutes, prior_per90=0.0,
                                              prior_minutes=prior_exposure)

    action_total = 0.0
    action_rows = 0
    for row in played:
        value = row.external_defensive_action_sum(position, rules)
        if value is not None:
            action_total += float(value)
            action_rows += 1
    if action_rows and minutes > 0:
        # prior_action_rate is PER 90, matching the feature's own units
        rate = shrunken_per90(total=action_total, minutes=minutes, prior_per90=prior_action_rate,
                              prior_minutes=prior_exposure)
        predictors["defensive_action_rate"] = rate
        predictors["defcon_per90"] = rate
        projected_defcon = rate * (float(expected_minutes) / 90.0)
    else:
        rate = None
        projected_defcon = None

    if opponent is not None:
        predictors["opponent_possession"] = opponent.value("possession")
        shots = opponent.value("total_shots")
        box = opponent.value("touches_in_opposition_box")
        # a simple, auditable pressure score: opponent shots and box touches
        predictors["opponent_attacking_pressure"] = (
            None if (shots is None and box is None)
            else float(_zero_if_none(shots)) + 0.1 * float(_zero_if_none(box))
        )

    p_threshold = None
    expected_points = None
    variance = None
    if projected_defcon is not None and threshold is not None:
        p_threshold = poisson_tail(projected_defcon, int(threshold))
        expected_points = float(rules.defcon_points) * p_threshold
        variance = float(rules.defcon_points) ** 2 * p_threshold * (1.0 - p_threshold)

    return DefconFeatures(
        official_player_id=int(played[0].official_player_id or -1) if played else -1,
        position=str(position).upper(),
        threshold=threshold,
        posterior_action_rate=rate,
        projected_defcon=projected_defcon,
        p_defcon_threshold=p_threshold,
        expected_defcon_points=expected_points,
        defcon_variance=variance,
        predictors=predictors,
        minutes=minutes,
        matches_used=len(played),
    )
