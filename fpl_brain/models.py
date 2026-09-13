"""Plain records exchanged between parsing and persistence layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Raw = dict[str, Any]


@dataclass
class EventRecord:
    id: int
    name: str | None = None
    deadline_time: str | None = None
    deadline_time_epoch: int | None = None
    finished: int | None = None
    data_checked: int | None = None
    is_previous: int | None = None
    is_current: int | None = None
    is_next: int | None = None
    average_entry_score: int | None = None
    highest_score: int | None = None
    most_selected: int | None = None
    most_transferred_in: int | None = None
    most_captained: int | None = None
    most_vice_captained: int | None = None
    top_element: int | None = None
    transfers_made: int | None = None
    released: int | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class ChipRecord:
    id: int
    name: str
    number: int | None = None
    chip_type: str | None = None
    start_event: int | None = None
    stop_event: int | None = None


@dataclass
class TeamRecord:
    id: int
    code: int | None = None
    name: str | None = None
    short_name: str | None = None
    strength: int | None = None
    strength_overall_home: int | None = None
    strength_overall_away: int | None = None
    strength_attack_home: int | None = None
    strength_attack_away: int | None = None
    strength_defence_home: int | None = None
    strength_defence_away: int | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class PositionRecord:
    id: int
    singular_name: str | None = None
    singular_name_short: str | None = None
    plural_name: str | None = None
    squad_select: int | None = None
    squad_min_play: int | None = None
    squad_max_play: int | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class PlayerRecord:
    id: int
    code: int | None = None
    first_name: str | None = None
    second_name: str | None = None
    web_name: str | None = None
    full_name: str | None = None
    norm_name: str | None = None
    team_id: int | None = None
    element_type: int | None = None
    squad_number: int | None = None
    opta_code: str | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class PlayerSnapshotRecord:
    player_id: int
    captured_at: str
    event_context: int | None = None
    now_cost: int | None = None
    cost_change_event: int | None = None
    cost_change_start: int | None = None
    selected_by_percent: float | None = None
    transfers_in_event: int | None = None
    transfers_out_event: int | None = None
    transfers_in: int | None = None
    transfers_out: int | None = None
    status: str | None = None
    news: str | None = None
    news_added: str | None = None
    chance_of_playing_this_round: int | None = None
    chance_of_playing_next_round: int | None = None
    total_points: int | None = None
    event_points: int | None = None
    points_per_game: float | None = None
    form: float | None = None
    minutes: int | None = None
    starts: int | None = None
    goals_scored: int | None = None
    assists: int | None = None
    clean_sheets: int | None = None
    goals_conceded: int | None = None
    saves: int | None = None
    bonus: int | None = None
    bps: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    penalties_saved: int | None = None
    penalties_missed: int | None = None
    influence: float | None = None
    creativity: float | None = None
    threat: float | None = None
    ict_index: float | None = None
    expected_goals: float | None = None
    expected_assists: float | None = None
    expected_goal_involvements: float | None = None
    expected_goals_conceded: float | None = None
    expected_goals_per_90: float | None = None
    expected_assists_per_90: float | None = None
    expected_goal_involvements_per_90: float | None = None
    expected_goals_conceded_per_90: float | None = None
    defensive_contribution: int | None = None
    clearances_blocks_interceptions: int | None = None
    recoveries: int | None = None
    tackles: int | None = None
    ep_this: float | None = None
    ep_next: float | None = None
    value_form: float | None = None
    value_season: float | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class FixtureRecord:
    id: int
    code: int | None = None
    event: int | None = None
    kickoff_time: str | None = None
    team_h: int | None = None
    team_a: int | None = None
    team_h_score: int | None = None
    team_a_score: int | None = None
    team_h_difficulty: int | None = None
    team_a_difficulty: int | None = None
    started: int | None = None
    finished: int | None = None
    finished_provisional: int | None = None
    provisional_start_time: int | None = None
    minutes: int | None = None
    stats_json: Any = field(default_factory=list)
    raw_json: Raw = field(default_factory=dict)


@dataclass
class PlayerGameweekRecord:
    player_id: int
    event: int
    fixture_id: int | None = None
    opponent_team: int | None = None
    was_home: int | None = None
    kickoff_time: str | None = None
    minutes: int | None = None
    starts: int | None = None
    total_points: int | None = None
    goals_scored: int | None = None
    assists: int | None = None
    clean_sheets: int | None = None
    goals_conceded: int | None = None
    saves: int | None = None
    bonus: int | None = None
    bps: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    penalties_saved: int | None = None
    penalties_missed: int | None = None
    own_goals: int | None = None
    influence: float | None = None
    creativity: float | None = None
    threat: float | None = None
    ict_index: float | None = None
    expected_goals: float | None = None
    expected_assists: float | None = None
    expected_goal_involvements: float | None = None
    expected_goals_conceded: float | None = None
    defensive_contribution: int | None = None
    value: int | None = None
    selected: int | None = None
    transfers_in: int | None = None
    transfers_out: int | None = None
    transfers_balance: int | None = None
    source: str = "element_summary"
    raw_json: Raw = field(default_factory=dict)


@dataclass
class BootstrapRecords:
    events: list[EventRecord]
    chips: list[ChipRecord]
    teams: list[TeamRecord]
    positions: list[PositionRecord]
    players: list[PlayerRecord]
    snapshots: list[PlayerSnapshotRecord]


@dataclass
class EntryRecord:
    entry_id: int
    player_name: str | None = None
    team_name: str | None = None
    summary_overall_points: int | None = None
    summary_overall_rank: int | None = None
    summary_event_points: int | None = None
    summary_event_rank: int | None = None
    current_event: int | None = None
    bank: int | None = None
    team_value: int | None = None
    total_transfers: int | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class HistoryRow:
    event: int | None = None
    bank: int | None = None
    value: int | None = None
    total_transfers: int | None = None
    event_transfers: int | None = None
    event_transfers_cost: int | None = None
    points_on_bench: int | None = None
    overall_rank: int | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class ManagerChipRecord:
    name: str
    event: int
    time: str | None = None


@dataclass
class ManagerHistory:
    current: list[HistoryRow] = field(default_factory=list)
    past: list[HistoryRow] = field(default_factory=list)
    chips: list[ManagerChipRecord] = field(default_factory=list)


@dataclass
class PickRecord:
    player_id: int
    position: int
    multiplier: int | None = None
    is_captain: int | None = None
    is_vice_captain: int | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class PicksRecord:
    picks: list[PickRecord] = field(default_factory=list)
    entry_history: HistoryRow | None = None
    active_chip: str | None = None
    automatic_subs: list[dict[str, Any]] = field(default_factory=list)
    raw_json: Raw = field(default_factory=dict)


@dataclass
class ManagerTransferRecord:
    """One exact transfer-history row from the public manager endpoint."""

    entry_id: int
    element_in: int
    element_out: int
    event: int
    time: str | None = None
    element_in_cost: int | None = None
    element_out_cost: int | None = None
    raw_json: Raw = field(default_factory=dict)


@dataclass
class PlayerSeasonHistoryRecord:
    """One official element-summary `history_past` season row.

    Official fields proven present: `season_name`, `starts`, `minutes`
    (plus scoring/identity context).  Appearances are NOT an official field;
    any match-count estimate downstream needs an explicit modelling
    assumption rather than pretending this row carries one.
    """

    player_id: int
    season_name: str
    minutes: int | None = None
    starts: int | None = None
    total_points: int | None = None
    goals_scored: int | None = None
    assists: int | None = None
    clean_sheets: int | None = None
    bonus: int | None = None
    saves: int | None = None
    raw_json: Raw = field(default_factory=dict)
