import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpl_brain.chip_horizon_evidence_assembly import (
    FutureStateAssemblyError,
    NO_TRANSFER_CONTINUATION_MODEL,
    _build_canonical_roll_route,
    _build_future_save_route,
    _add_future_lineup_policies,
    _project_future_manager_state,
    load_projected_future_manager_state,
)
from fpl_brain.chip_route_assembly import VerifiedNormalRoute
from fpl_brain.route_optimizer import PartialRoute
from fpl_brain.season_rules import SeasonRules
from fpl_brain import transfer_state as ts
from fpl_brain import manager_lineup as ml
from fpl_brain.transfer_state import PlayerMeta, RoutePlayer, RouteState


RULES = SeasonRules(season="2026/27", max_free_transfers=5)


def _route(*, origin_event=3, event_start_ft=2, transfer_at_event=None):
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = tuple(
        RoutePlayer(
            player_id=pid,
            position=position,
            club_id=((pid - 1) // 3) + 1,
            purchase_price_tenths=50,
        )
        for pid, position in enumerate(positions, start=1)
    )
    player_meta = {
        player.player_id: PlayerMeta(player.player_id, player.position, player.club_id)
        for player in players
    }
    player_meta[16] = PlayerMeta(16, "MID", 6)
    prices = {pid: 50 for pid in player_meta}
    prices[16] = 55
    state = RouteState(
        event=origin_event,
        players=players,
        bank_tenths=7,
        free_transfers=2,
        chip_state=(),
        event_start_free_transfers=event_start_ft,
    )
    initial = state
    actions = []
    events = tuple(range(origin_event, origin_event + 4))
    for event in events:
        batch = (
            ts.TransferBatch((ts.TransferAction(8, 16),))
            if int(transfer_at_event or -1) == event
            else ts.TransferBatch.roll()
        )
        transition = ts.apply_transfer_batch(
            state, batch, ts.PriceSnapshot(event=event, prices=prices), player_meta,
        )
        assert transition.ok, transition.errors
        next_state = transition.next_event_state
        actions.append({
            "event": event,
            "kind": "SINGLE" if batch.actions else "ROLL",
            "batch": batch,
            "transition": transition,
            "hit_points": transition.hit_points,
            "delta_3gw": 0.0,
            "squad_ids": tuple(sorted(next_state.by_id())),
            "ft_after": next_state.free_transfers,
            "bank_after": next_state.bank_tenths,
        })
        state = next_state
    partial = PartialRoute(
        state=state, actions=tuple(actions), h1_proxy=0.0, window_proxy=0.0, hits=0,
    )
    return VerifiedNormalRoute(
        source_decision_id="decision-1",
        source_result_sha256="a" * 64,
        source_artifact_sha256="b" * 64,
        source_artifact_ref="unused-fixture-artifact.json",
        generation_id="generation-1",
        planning_event=origin_event,
        events=events,
        cutoff="2026-10-04T08:00:00Z",
        data_snapshot_sha256="c" * 64,
        certification_identity="d" * 64,
        route_id="route-fixture",
        actual_owned_ids=tuple(range(1, 16)),
        proposed_owned_ids=tuple(range(1, 16)),
        post_h1_bank_tenths=7,
        post_h1_free_transfers=3,
        policy=None,
        positions={pid: row.position for pid, row in player_meta.items()},
        clubs={pid: row.club_id for pid, row in player_meta.items()},
        partial=partial,
        route_input_sha256="e" * 64,
        runner_code_identity="fixture-only",
    ), initial


def _make_verified_origin(
    tmp_path, monkeypatch, *, event_start_ft, include_event8=False, extended_events=(),
    chip_expiry_event=5, free_hit_expiry_event=None, free_transfers=2,
    later_paid_hit_event=None, paid_fh_play_event=None,
    historical_paid_hit_event=None, suppress_fh_play_standouts=False,
):
    """Build a real, tiny PE-9 generation/decision/route with an event-5 transfer."""
    import generation_fixtures as gf
    import test_pe9_production_decision as pe9
    from fpl_brain import (
        candidate_universe as cu,
        chip_route_assembly as cra,
        generation_store as gs,
        ingest_provenance,
        manager_lineup as ml,
        raw_archive,
        repositories as repo,
        route_comparator as rc,
        transfer_state as transfer,
    )

    original_prepare = gf.prepare_fixture_snapshot
    raw_root = tmp_path / "origin-raw"
    standout_mid_ids: list[int] = []
    expanded_fixture_inputs = any(value is not None for value in (
        later_paid_hit_event, paid_fh_play_event, historical_paid_hit_event,
    ))
    if paid_fh_play_event is not None:
        assert int(paid_fh_play_event) == 6
    if historical_paid_hit_event is not None:
        assert int(historical_paid_hit_event) == 4
    fh_expiry_event = (
        int(chip_expiry_event) if free_hit_expiry_event is None else int(free_hit_expiry_event)
    )

    def prepare_with_archived_rules(conn, events, *, cutoff=gf.CUTOFF):
        raw_observed_at = "2026-09-19T10:50:00Z"
        run_id = repo.create_fetch_run(
            conn,
            "fetch_fpl",
            raw_dir=str(raw_root),
            started_at="2026-09-19T10:49:00Z",
        )
        repo.finish_fetch_run(
            conn, run_id, "success", current_event=4, endpoints_ok=["bootstrap-static"],
        )
        conn.execute(
            "UPDATE fetch_runs SET finished_at=? WHERE id=?",
            (raw_observed_at, int(run_id)),
        )
        if expanded_fixture_inputs:
            if later_paid_hit_event is not None:
                assert int(later_paid_hit_event) == 6
            from fpl_brain.models import PlayerRecord, PlayerSnapshotRecord

            repo.upsert_players(conn, [
                PlayerRecord(id=44, web_name="P44", full_name="Player 44", team_id=40, element_type=2),
                PlayerRecord(id=45, web_name="P45", full_name="Player 45", team_id=1, element_type=3),
            ])
            repo.insert_snapshots(conn, [
                PlayerSnapshotRecord(player_id=44, captured_at=raw_observed_at, now_cost=45, raw_json={}),
                PlayerSnapshotRecord(player_id=45, captured_at=raw_observed_at, now_cost=45, raw_json={}),
            ], run_id)
            # Complete the two synthetic fixture sides before the accepted
            # bootstrap snapshot and generation are certified: one keeper and
            # ten outfielders per side, including P45 in GW5.
            synthetic_specs = (
                (1, ((2, 4), (3, 3), (4, 2))),
                (2, ((2, 4), (3, 4), (4, 2))),
            )
            synthetic_players = []
            synthetic_snapshots = []
            synthetic_id = 1000
            for team_id, position_counts in synthetic_specs:
                for element_type, count in position_counts:
                    for _ in range(count):
                        player_id = synthetic_id
                        synthetic_id += 1
                        synthetic_players.append(PlayerRecord(
                            id=player_id, web_name=f"S{player_id}",
                            full_name=f"Synthetic Player {player_id}",
                            team_id=team_id, element_type=element_type,
                        ))
                        synthetic_snapshots.append(PlayerSnapshotRecord(
                            player_id=player_id, captured_at=raw_observed_at,
                            now_cost=45, raw_json={},
                        ))
            repo.upsert_players(conn, synthetic_players)
            repo.insert_snapshots(conn, synthetic_snapshots, run_id)
            # Complete every supplemental match side before the accepted
            # bootstrap pool and fixture snapshot are bound. Each side is a
            # coherent 1-GKP / 10-outfielder simulation population.
            supplemental_rosters = (
                (11, ((1, 1), (3, 4), (4, 3))),
                (12, ((1, 1), (2, 1), (3, 3), (4, 3))),
                (14, ((1, 1), (2, 3), (3, 3), (4, 1))),
                (15, ((1, 1), (2, 3), (3, 4), (4, 2))),
            )
            supplemental_players = []
            supplemental_snapshots = []
            supplemental_id = 2000
            for team_id, position_counts in supplemental_rosters:
                for element_type, count in position_counts:
                    for ordinal in range(count):
                        player_id = supplemental_id
                        supplemental_id += 1
                        supplemental_players.append(PlayerRecord(
                            id=player_id, web_name=f"R{player_id}",
                            full_name=f"Roster Player {player_id}",
                            team_id=team_id, element_type=element_type,
                        ))
                        supplemental_snapshots.append(PlayerSnapshotRecord(
                            player_id=player_id, captured_at=raw_observed_at,
                            now_cost=40, raw_json={},
                        ))
                        if team_id == 15 and element_type == 3 and ordinal < 2:
                            standout_mid_ids.append(player_id)
            repo.upsert_players(conn, supplemental_players)
            repo.insert_snapshots(conn, supplemental_snapshots, run_id)
        element_ids = repo.active_player_ids(conn)
        repo.record_bootstrap_generation(
            conn,
            captured_at=raw_observed_at,
            accepted=True,
            official_element_count=len(element_ids),
            parsed_count=len(element_ids),
            persisted_count=len(element_ids),
            element_ids=element_ids,
            element_ids_sha256=ingest_provenance.element_id_sha256(element_ids),
            acceptance_rule="tiny integration fixture exact active pool",
            acceptance_rule_version="fixture-v1",
            fetch_run_id=run_id,
        )
        bootstrap = {
            "game_settings": {
                "squad_squadsize": 15,
                "squad_squadplay": 11,
                "squad_team_limit": 3,
                "squad_total_spend": 1000,
                "max_extra_free_transfers": 4,
                "transfers_sell_on_fee": 0.5,
                "element_sell_at_purchase_price": False,
            },
        }
        raw_archive.archive_raw_capture(
            raw_root,
            source="bootstrap_static",
            observed_at=raw_observed_at,
            body=json.dumps(bootstrap, separators=(",", ":")).encode("utf-8"),
            run_id=run_id,
        )
        snapshot_events = tuple(int(event) for event in events)
        future_events = tuple(sorted({int(event) for event in extended_events}))
        if include_event8:
            future_events = tuple(sorted(set(future_events) | {8}))
        if future_events:
            for event in future_events:
                gf.add_event(conn, event)
                gf.add_fixture(conn, 45 + (event - 8), event, 1, 2)
            from fpl_brain import chip_decision as cd

            conn.execute(
                "UPDATE chip_definitions SET start_event=4,stop_event=?,updated_at=? WHERE name=?",
                (int(chip_expiry_event), raw_observed_at,
                 cd.CHIP_ACTION_TO_OFFICIAL_NAME[cd.CHIP_ACTION_WC]),
            )
            conn.execute(
                "INSERT INTO chip_definitions(id,name,number,chip_type,start_event,stop_event,updated_at)"
                " VALUES (2,?,1,'transfer',4,?,?)",
                (cd.CHIP_ACTION_TO_OFFICIAL_NAME[cd.CHIP_ACTION_FH],
                 fh_expiry_event, raw_observed_at),
            )
            snapshot_events = (*snapshot_events, *future_events)
        if expanded_fixture_inputs:
            # Fixture IDs are separate from the existing 41-48 IDs. The same
            # three fixture pairs then exist in every certified future event.
            for event in sorted({int(value) for value in snapshot_events if int(value) >= 5}):
                gf.add_fixture(conn, 1000 + 2 * event, event, 11, 12)
                gf.add_fixture(conn, 1001 + 2 * event, event, 14, 15)
                fixture_pairs = {
                    (int(row["team_h"]), int(row["team_a"]))
                    for row in conn.execute(
                        "SELECT team_h,team_a FROM fixtures WHERE event=?", (event,),
                    )
                }
                assert fixture_pairs == {(1, 2), (11, 12), (14, 15)}
            for team_id in (1, 2, 11, 12, 14, 15):
                counts = {
                    int(row["element_type"]): int(row["n"])
                    for row in conn.execute(
                        "SELECT element_type,COUNT(*) AS n FROM players "
                        "WHERE team_id=? GROUP BY element_type", (team_id,),
                    )
                }
                assert sum(counts.values()) == 11, (team_id, counts)
                assert counts.get(1, 0) == 1, (team_id, counts)
                assert sum(value for position, value in counts.items() if position != 1) == 10
            for team_id in (11, 12, 14, 15):
                counts = {
                    int(row["element_type"]): int(row["n"])
                    for row in conn.execute(
                        "SELECT element_type,COUNT(*) AS n FROM players "
                        "WHERE team_id=? GROUP BY element_type", (team_id,),
                    )
                }
                assert counts == {1: 1, 2: 3, 3: 4, 4: 3}, (team_id, counts)
        conn.commit()
        return original_prepare(conn, snapshot_events, cutoff=cutoff)

    monkeypatch.setattr(gf, "prepare_fixture_snapshot", prepare_with_archived_rules)
    if include_event8 or extended_events or expanded_fixture_inputs:
        from fpl_brain import analytics, joint_minutes, player_rates, xpts as xpts_model
        from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES

        def add_complete_fixture_process_rows(
            conn, run_id, fixture_id, event, *, minutes_run_id, team_run_id, rate_run_id,
            player_id=1,
        ):
            del player_id
            fixture = conn.execute(
                "SELECT team_h,team_a FROM fixtures WHERE id=?", (int(fixture_id),)
            ).fetchone()
            teams = {int(fixture["team_h"]), int(fixture["team_a"])}
            positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
            fixture_players = [dict(row) for row in conn.execute(
                "SELECT id,team_id,element_type FROM players WHERE team_id IN (?,?) ORDER BY id",
                tuple(sorted(teams)),
            )]
            complete_mc_fixture = expanded_fixture_inputs
            xg_by_player = {}
            if complete_mc_fixture:
                xg_by_player = {
                    int(player["id"]): (
                        8.0 if int(event) == 5 and int(player["id"]) == 45
                        else 100.0 if (
                            paid_fh_play_event is not None
                            and (
                                (int(player["id"]) == 2001 and int(event) >= int(paid_fh_play_event))
                                or (int(player["id"]) in {2002, 2028, 2029} and int(event) >= int(paid_fh_play_event))
                            )
                        )
                        else 0.0 if int(player["element_type"]) == 1
                        else 0.05 if (
                            suppress_fh_play_standouts
                            and int(player["id"]) in standout_mid_ids
                        )
                        else 0.5 if int(player["id"]) in standout_mid_ids
                        else 0.05
                    )
                    for player in fixture_players
                }
                xg_by_team = {
                    team: sum(
                        xg_by_player[int(player["id"])]
                        for player in fixture_players if int(player["team_id"]) == team
                    )
                    for team in teams
                }
                for team in sorted(teams):
                    opponent = next(value for value in teams if value != team)
                    analytics.freeze_team_fixture_projection(
                        conn, int(team_run_id), fixture_id=int(fixture_id), event=int(event),
                        team_id=team, opponent_id=opponent,
                        venue="home" if team == int(fixture["team_h"]) else "away",
                        payload={
                            "expected_goals_for": xg_by_team[team],
                            "expected_goals_against": xg_by_team[opponent],
                        },
                        model_version="team_strength_v1.1.0",
                        generated_at="2026-09-19T11:01:00Z",
                    )
            for player in fixture_players:
                player_id = int(player["id"])
                team_id = int(player["team_id"])
                opponent_id = next(value for value in teams if value != team_id)
                position = positions[int(player["element_type"])]
                if complete_mc_fixture:
                    xg = xg_by_player[player_id]
                    goal_xpts = xg * DEFAULT_SCORING_RULES.goal_points_for(position)
                    payload = {
                        "core_xpts": 2.0 + goal_xpts,
                        "expected_minutes": 90.0,
                        "p_start": 1.0,
                        "p_appearance": 1.0,
                        "p_60_plus": 1.0,
                        "fixture_xg_per90": xg,
                        "fixture_xa_per90": 0.0,
                        "raw_expected_xg": xg,
                        "adjusted_expected_xg": xg,
                        "xg_scale_factor": 1.0,
                        "expected_xa": 0.0,
                        "expected_fpl_assists": 0.0,
                        "appearance_xpts": 2.0,
                        "goal_xpts": goal_xpts,
                        "assist_xpts": 0.0,
                        "clean_sheet_xpts": 0.0,
                        "goals_conceded_xpts": 0.0,
                        "defcon_xpts": 0.0,
                        "save_xpts": 0.0,
                        "yellow_card_xpts": 0.0,
                        "bonus_xpts": 0.0,
                    }
                    for component, rate in (
                        (player_rates.COMPONENT_XG, xg),
                        (player_rates.COMPONENT_XA, 0.0),
                    ):
                        analytics.freeze_player_rate_projection(
                            conn, int(rate_run_id), player_id=player_id,
                            component=component, event=int(event),
                            payload={"posterior_mean": rate},
                            model_version=player_rates.PLAYER_RATE_MODEL_VERSION,
                            generated_at="2026-09-19T11:01:00Z",
                        )
                    minute_payload = {
                        "joint_start_target": 1.0,
                        "joint_availability": 1.0,
                        "joint_exit_propensity": 0.0,
                        "joint_entry_propensity": 0.0,
                        "joint_position": position,
                        "joint_expected_minutes_if_cameo": 0.0,
                        "primitive_source_version": joint_minutes.JOINT_MINUTES_MODEL_VERSION,
                        "expected_minutes": 90.0,
                    }
                    minutes_version = joint_minutes.JOINT_MINUTES_MODEL_VERSION
                else:
                    payload = {
                        "core_xpts": 4.5,
                        "expected_minutes": 76.0,
                        "p_start": 0.82,
                        "p_60_plus": 0.78,
                    }
                    minute_payload = {"joint_availability": 0.94, "expected_minutes": 76.0}
                    minutes_version = "fixture-minutes-v1"
                conn.execute(
                    "INSERT INTO player_fixture_xpts_projections("
                    "projection_run_id,player_id,fixture_id,event,team_id,opponent_id,position,"
                    "minutes_run_id,team_run_id,rate_run_id,payload_json,model_version,"
                    "scoring_rules_version,generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        int(run_id), player_id, int(fixture_id), int(event), team_id, opponent_id,
                        position, int(minutes_run_id), int(team_run_id), int(rate_run_id),
                        json.dumps(payload, sort_keys=True), "xpts_v1", "v1",
                        "2026-09-19T11:01:00Z",
                    ),
                )
                analytics.freeze_prediction(
                    conn,
                    int(minutes_run_id),
                    kind=analytics.MINUTES_V1_KIND,
                    player_id=player_id,
                    event=int(event),
                    fixture_id=int(fixture_id),
                    payload=minute_payload,
                    model_version=minutes_version,
                    generated_at="2026-09-19T11:01:00Z",
                )

        if not expanded_fixture_inputs:
            monkeypatch.setattr(gf, "add_xpts_row", add_complete_fixture_process_rows)
        else:
            def event_fixture_ids(conn, event):
                return [
                    int(row["id"]) for row in conn.execute(
                        "SELECT id FROM fixtures WHERE event=? ORDER BY id", (int(event),),
                    )
                ]

            def assert_complete_event_projection_inputs(
                conn, *, event, xpts_run_id, minutes_run_id, team_run_id,
                rate_run_id, mc_run_id,
            ):
                fixtures = [
                    dict(row) for row in conn.execute(
                        "SELECT id,team_h,team_a FROM fixtures WHERE event=? ORDER BY id",
                        (int(event),),
                    )
                ]
                expected_ids = {int(row["id"]) for row in fixtures}
                assert expected_ids
                xpts_counts = {
                    int(row["fixture_id"]): int(row["n"])
                    for row in conn.execute(
                        "SELECT fixture_id,COUNT(*) AS n FROM player_fixture_xpts_projections "
                        "WHERE projection_run_id=? GROUP BY fixture_id", (int(xpts_run_id),),
                    )
                }
                minutes_counts = {
                    int(row["fixture_id"]): int(row["n"])
                    for row in conn.execute(
                        "SELECT fixture_id,COUNT(*) AS n FROM frozen_predictions "
                        "WHERE projection_run_id=? AND kind=? AND event=? "
                        "GROUP BY fixture_id",
                        (int(minutes_run_id), analytics.MINUTES_V1_KIND, int(event)),
                    )
                }
                team_counts = {
                    int(row["fixture_id"]): int(row["n"])
                    for row in conn.execute(
                        "SELECT fixture_id,COUNT(*) AS n FROM team_fixture_projections "
                        "WHERE projection_run_id=? GROUP BY fixture_id", (int(team_run_id),),
                    )
                }
                mc_ids = {
                    int(row["fixture_id"]) for row in conn.execute(
                        "SELECT DISTINCT fixture_id FROM monte_carlo_distributions "
                        "WHERE projection_run_id=?", (int(mc_run_id),),
                    )
                }
                assert set(xpts_counts) == set(minutes_counts) == set(team_counts) == mc_ids == expected_ids
                expected_rate_keys = set()
                for fixture in fixtures:
                    fixture_players = {
                        int(row["id"]) for row in conn.execute(
                            "SELECT id FROM players WHERE team_id IN (?,?)",
                            (int(fixture["team_h"]), int(fixture["team_a"])),
                        )
                    }
                    assert len(fixture_players) == 22
                    assert xpts_counts[int(fixture["id"])] == 22
                    assert minutes_counts[int(fixture["id"])] == 22
                    assert team_counts[int(fixture["id"])] == 2
                    for player_id in fixture_players:
                        expected_rate_keys.add((player_id, player_rates.COMPONENT_XG))
                        expected_rate_keys.add((player_id, player_rates.COMPONENT_XA))
                actual_rate_keys = {
                    (int(row["player_id"]), str(row["component"]))
                    for row in conn.execute(
                        "SELECT player_id,component FROM player_rate_projections "
                        "WHERE projection_run_id=? AND event=?",
                        (int(rate_run_id), int(event)),
                    )
                }
                assert actual_rate_keys == expected_rate_keys

            projected_xpts: set[tuple[int, int]] = set()
            projected_mc: set[tuple[int, int]] = set()

            def add_complete_event_process_rows(
                conn, run_id, fixture_id, event, *, minutes_run_id, team_run_id,
                rate_run_id, player_id=1,
            ):
                del fixture_id, player_id
                fixture_ids = event_fixture_ids(conn, event)
                assert fixture_ids
                for target_fixture_id in fixture_ids:
                    key = (int(run_id), int(target_fixture_id))
                    if key in projected_xpts:
                        continue
                    add_complete_fixture_process_rows(
                        conn, run_id, target_fixture_id, event,
                        minutes_run_id=minutes_run_id, team_run_id=team_run_id,
                        rate_run_id=rate_run_id,
                    )
                    projected_xpts.add(key)

            def add_complete_event_mc_rows(
                conn, run_id, fixture_id, event, *, xpts_run_id, minutes_run_id,
                team_run_id, rate_run_id, player_id=1,
            ):
                del fixture_id, player_id
                fixture_ids = event_fixture_ids(conn, event)
                assert fixture_ids
                position_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
                for target_fixture_id in fixture_ids:
                    key = (int(run_id), int(target_fixture_id))
                    if key in projected_mc:
                        continue
                    fixture = conn.execute(
                        "SELECT team_h,team_a FROM fixtures WHERE id=?",
                        (target_fixture_id,),
                    ).fetchone()
                    home_player = conn.execute(
                        "SELECT id,element_type FROM players WHERE team_id=? ORDER BY id LIMIT 1",
                        (int(fixture["team_h"]),),
                    ).fetchone()
                    analytics.freeze_monte_carlo_distribution(
                        conn, int(run_id), player_id=int(home_player["id"]),
                        fixture_id=target_fixture_id, event=int(event),
                        team_id=int(fixture["team_h"]), opponent_id=int(fixture["team_a"]),
                        position=position_names[int(home_player["element_type"])],
                        xpts_run_id=int(xpts_run_id), minutes_run_id=int(minutes_run_id),
                        team_run_id=int(team_run_id), rate_run_id=int(rate_run_id),
                        payload={}, model_version="mc_v1",
                        generated_at="2026-09-19T11:01:00Z",
                    )
                    projected_mc.add(key)
                assert_complete_event_projection_inputs(
                    conn, event=event, xpts_run_id=xpts_run_id,
                    minutes_run_id=minutes_run_id, team_run_id=team_run_id,
                    rate_run_id=rate_run_id, mc_run_id=run_id,
                )

            monkeypatch.setattr(gf, "add_xpts_row", add_complete_event_process_rows)
            monkeypatch.setattr(gf, "add_mc_row", add_complete_event_mc_rows)
    db_path = tmp_path / "origin.db"
    conn, generation = pe9._four_gw_certified_manager_world(
        db_path,
        bank=7,
        free_transfers=int(free_transfers),
        event_start_free_transfers=event_start_ft,
    )
    future_events = tuple(sorted({int(event) for event in extended_events}))
    if include_event8:
        future_events = tuple(sorted(set(future_events) | {8}))
    for index, event in enumerate(future_events):
        ids = {
            "minutes_v1": 120 + index * 5,
            "team_strength_v1": 121 + index * 5,
            "player_rates_v1": 122 + index * 5,
            "xpts_v1": 123 + index * 5,
            "monte_carlo_v1": 124 + index * 5,
        }
        with conn:
            for family, run_id in ids.items():
                gf.add_run(conn, run_id, family, event, cutoff=generation.cutoff)
            gf.add_xpts_row(
                conn, ids["xpts_v1"], 45 + (event - 8), event,
                minutes_run_id=ids["minutes_v1"],
                team_run_id=ids["team_strength_v1"],
                rate_run_id=ids["player_rates_v1"],
            )
            gf.add_mc_row(
                conn, ids["monte_carlo_v1"], 45 + (event - 8), event,
                xpts_run_id=ids["xpts_v1"],
                minutes_run_id=ids["minutes_v1"],
                team_run_id=ids["team_strength_v1"],
                rate_run_id=ids["player_rates_v1"],
            )

    def canonical_route_runner(**kwargs):
        canonical = kwargs["canonical_manager_state"]
        source_conn = kwargs["source_conn"]
        gen = kwargs["generation"]
        initial = cra._state_from_payload(canonical["route_state"])
        candidate_ids = {43}
        if later_paid_hit_event is not None:
            candidate_ids.add(44)
        if historical_paid_hit_event is not None:
            candidate_ids.add(2001)
        relevant_ids = sorted(set(initial.by_id()) | candidate_ids)
        price_snapshot = cu.price_snapshot_as_of(
            source_conn, gen.planning_event, gen.cutoff, required_player_ids=relevant_ids,
        )
        scenario = rc.flat_current_price_scenario(price_snapshot, gen.events)
        meta = rc.load_player_meta(source_conn, relevant_ids)
        policy = ml.ManagerPolicy(
            starter_ids=(1, 13, 14, 15, 21, 22, 23, 24, 31, 32, 33),
            bench_gk_id=2,
            bench_outfield_order=(41, 42, 25),
            captain_id=21,
            vice_captain_id=22,
        )
        actions = []
        event_policies = []
        state = initial
        for event in gen.events:
            if historical_paid_hit_event is not None and int(event) == int(historical_paid_hit_event):
                batch = transfer.TransferBatch((transfer.TransferAction(25, 2001),))
            elif int(event) == 5:
                batch = transfer.TransferBatch((transfer.TransferAction(13, 43),))
            elif later_paid_hit_event is not None and int(event) == int(later_paid_hit_event):
                batch = transfer.TransferBatch((
                    transfer.TransferAction(43, 13), transfer.TransferAction(14, 44),
                ))
            else:
                batch = transfer.TransferBatch.roll()
            transition = transfer.apply_transfer_batch(
                state, batch, scenario.snapshot_for(event), meta,
            )
            assert transition.ok, transition.errors
            state = transition.next_event_state
            event_policy = policy
            if batch.actions:
                replacements = {
                    int(row.out_player_id): int(row.in_player_id) for row in batch.actions
                }
                event_policy = ml.ManagerPolicy(
                    starter_ids=tuple(replacements.get(int(pid), int(pid)) for pid in policy.starter_ids),
                    bench_gk_id=replacements.get(policy.bench_gk_id, policy.bench_gk_id),
                    bench_outfield_order=tuple(
                        replacements.get(int(pid), int(pid)) for pid in policy.bench_outfield_order
                    ),
                    captain_id=replacements.get(policy.captain_id, policy.captain_id),
                    vice_captain_id=replacements.get(policy.vice_captain_id, policy.vice_captain_id),
                )
                policy = event_policy
            event_policies.append({"event": int(event), "policy": event_policy.as_dict()})
            actions.append({
                "event": int(event),
                "kind": ("ROLL" if not batch.actions else "SINGLE" if len(batch.actions) == 1 else "DOUBLE"),
                "transfers": [
                    {"out": int(row.out_player_id), "in": int(row.in_player_id)}
                    for row in batch.actions
                ],
                "squad_ids": sorted(state.by_id()),
                "bank_after": state.bank_tenths,
                "ft_after": state.free_transfers,
                "hit_points": transition.hit_points,
            })
        route_row = {
            "actions": actions,
            "terminal_bank_tenths": state.bank_tenths,
            "terminal_ft": state.free_transfers,
            "per_event": event_policies,
        }
        return {
            "decision": {"status": "TINY_VERIFIED_ROUTE_FIXTURE"},
            "consumed_manager_state": canonical,
            "provenance": {"planning_context_hash": gs.four_gw_manager_context_identity(canonical)},
            "runner_identity": "tiny-verified-origin-fixture",
            "artifact_blocks": {"search": {"config": {"search_draws": 2, "seed": 62026}}},
            "finalist_refinement": {"route_table": {"routes": {"route_fixture": route_row}}},
        }

    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: canonical_route_runner)
    decision = gs.make_decision(
        conn,
        {
            "entry_id": 241392,
            "planning_event": 4,
            "cutoff": generation.cutoff,
            "season": "2026/27",
        },
        4,
        horizon_kind=gs.HORIZON_KIND_FOUR_GW,
        generation_id=generation.generation_id,
        profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
    )
    route_id = "route_fixture"
    return conn, generation, str(decision["decision_record_id"]), route_id


def _materialize_synthetic_fixture_caches(conn, generation, decision_id, cache_dir, *, events, windows):
    """Prepare genuine optimizer caches from certified temporary fixture runs.

    This setup-only helper is called before the public composer installs its
    no-simulation spy. It never touches operational data or cache directories.
    """
    from dataclasses import fields, replace
    from pathlib import Path
    from fpl_brain import candidate_universe as cu, generation_store as gs
    from fpl_brain import route_optimizer as ro
    from fpl_brain import chip_wildcard as cw

    record = gs.load_engine_decision_record(conn, decision_id)
    artifact = json.loads(Path(str(record["decision_artifact_ref"])).read_text(encoding="utf-8"))
    raw_config = dict((artifact.get("search") or {}).get("config") or {})
    allowed = {field.name for field in fields(ro.OptimizerConfig)}
    snapshot_conn = gs._open_generation_snapshot(generation)
    try:
        pool_binding = cw.pool_binding_from_store(snapshot_conn)
        pool = cu.load_pool(snapshot_conn)
    finally:
        snapshot_conn.close()
    player_ids = tuple(sorted(int(pid) for pid in pool_binding.eligible_ids))
    assert set(player_ids) == set(pool)
    cache_root = Path(cache_dir)
    materialized = {}
    for window in windows:
        window_events = tuple(int(event) for event in window)
        config_values = dict(raw_config)
        config_values["events"] = window_events
        config = ro.OptimizerConfig(**{
            key: value for key, value in config_values.items() if key in allowed
        })
        for event in events:
            matrix, cache_info = ro.build_event_worlds(
                conn,
                generation,
                int(event),
                player_ids,
                config,
                cache_dir=cache_root,
                allow_materialization=True,
            )
            materialized[(window_events, int(event))] = (matrix, cache_info)
    return materialized


def _assert_wc_value_greedy_seed_before_world_build(
    conn, *, decision_id, generation_id, route_id, entry_id, origin_cutoff,
    data_snapshot_sha256, target_event, season, value_generation,
):
    """Fast certified-projection preflight; intentionally contains no worlds."""
    from dataclasses import fields
    from fpl_brain import (
        analytics, candidate_universe as cu,
        chip_horizon_evidence_assembly as assembly, chip_wildcard as wc,
        free_hit_request_adapter as fa, generation_store as gs,
        route_optimizer as optimizer, transfer_state as ts,
        wildcard_production as wcp,
    )

    target = int(target_event)
    projected = assembly.load_projected_future_manager_state(
        conn, decision_id=str(decision_id), generation_id=str(generation_id),
        route_id=str(route_id), entry_id=int(entry_id), origin_cutoff=str(origin_cutoff),
        data_snapshot_sha256=str(data_snapshot_sha256), target_event=target,
        season=str(season),
    )
    events = tuple(int(event) for event in value_generation.events)
    snapshot_conn = gs._open_generation_snapshot(value_generation)
    try:
        pool = cu.load_pool(snapshot_conn)
        pool_binding = wc.pool_binding_from_store(snapshot_conn)
        assert set(pool) == set(pool_binding.eligible_ids)
        fixtures = cu.load_fixtures_by_team(snapshot_conn, events)
        prices = fa._canonical_market_prices(
            snapshot_conn, pool_binding.eligible_ids, str(value_generation.cutoff),
        )
        assert set(prices) == set(pool_binding.eligible_ids)
    finally:
        snapshot_conn.close()

    record = gs.load_engine_decision_record(conn, str(decision_id))
    artifact = json.loads(Path(str(record["decision_artifact_ref"])).read_text(encoding="utf-8"))
    raw_config = dict((artifact.get("search") or {}).get("config") or {})
    allowed = {field.name for field in fields(optimizer.OptimizerConfig)}
    raw_config["events"] = events
    config = optimizer.OptimizerConfig(**{
        key: value for key, value in raw_config.items() if key in allowed
    })
    model_config_identity = analytics.canonical_hash({
        "generation_id": value_generation.generation_id,
        "events": list(events),
        "runs_by_event": {
            str(event): value_generation.runs_for(event) for event in events
        },
        "world_config": config.as_dict(),
    })
    identity = wc.WildcardPredictiveIdentity(
        cutoff=str(value_generation.cutoff),
        data_snapshot_sha256=str(value_generation.snapshot.get("sha256") or ""),
        source_snapshot_sha256=str(value_generation.manifest.get("code_snapshot_sha256") or ""),
        generation=str(value_generation.generation_id),
        model_config_identity=model_config_identity,
    )
    xpts_by_event = {}
    minutes_by_event = {}
    for event in events:
        runs = value_generation.runs_for(event)
        xpts_by_event[event] = cu.load_projection_rows(conn, int(runs["xpts_v1"]))
        minutes_by_event[event] = assembly._load_minutes_projection_rows(
            conn, int(runs["minutes_v1"]),
        )
    players = wcp._players_from_certified_rows(
        pool=pool, prices=prices, events=events, fixtures=fixtures,
        xpts_by_event=xpts_by_event, minutes_by_event=minutes_by_event,
        identity=identity,
    )
    horizon = wcp.wildcard_horizon(
        target, length=len(events),
        last_event=int(value_generation.manifest.get("last_event") or events[-1]),
    )
    value_binding = wcp.value_horizon_binding(
        target, horizon, decision_cutoff=str(value_generation.cutoff),
        data_snapshot_sha256=str(value_generation.snapshot.get("sha256") or ""),
        source_snapshot_sha256=str(value_generation.manifest.get("code_snapshot_sha256") or ""),
        prediction_generation=str(value_generation.generation_id),
        model_config_identity=model_config_identity,
    )
    state = projected.state
    basis = {
        int(player.player_id): int(player.purchase_price_tenths)
        for player in state.players
    }
    selling = {
        int(pid): int(ts.selling_price_tenths(int(basis[int(pid)]), int(prices[int(pid)])))
        for pid in basis
    }
    # This request is deliberately incomplete for evaluation: it carries no
    # world matrices or chip binding and is used only by the canonical screen,
    # frontier and value-seed functions below.
    request = wc.WildcardRequest(
        planning_event=target, horizon=horizon, players=players,
        positions={int(pid): str(row.get("position") or "") for pid, row in pool.items()},
        owned_ids=tuple(sorted(basis)), purchase_price_tenths=basis,
        selling_price_tenths=selling, bank_tenths=int(state.bank_tenths),
        rules=projected.season_rules, horizon_binding=None,
        certification_identity="", data_snapshot_sha256=str(data_snapshot_sha256),
        value_horizon_binding=value_binding, pool_binding=pool_binding,
    )
    scores, screen = wc.screen_players(request)
    assert screen["screened"] == len(pool_binding.eligible_ids)
    standouts = [
        pid for pid, player in players.items()
        if player.club_id == 15 and player.position == "MID"
        and player.at(target).expected_points > 4.0
    ]
    assert len(standouts) == 2
    if target == 5:
        assert scores[45] > max(scores[pid] for pid in standouts)
    elif target == 6:
        assert scores[45] < min(scores[pid] for pid in standouts)
    else:
        raise AssertionError(f"unexpected preflight target GW{target}")
    frontiers = wc.position_frontiers(request, scores)
    seed = wc._greedy_squad(request, scores, frontiers, key="value")
    assert seed is not None, f"GW{target} canonical value frontier has no legal-size seed"
    legal, problems = wc.is_legal_squad(request, seed)
    assert legal, f"GW{target} canonical value seed is illegal: {problems}"
    return tuple(int(pid) for pid in seed)


def _archive_future_official_state(
    conn, raw_root, events, *, final_events=(), live_players_by_event=None, observed_after=None,
    observed_at_override=None, finished_at_override=None, wall_clock_base_override=None,
):
    """Create a completed, intact bootstrap/fixture source witness in the temp DB."""
    from datetime import datetime, timedelta, timezone
    from fpl_brain import raw_archive, repositories as repo

    targets = tuple(sorted({int(event) for event in events}))
    final_set = {int(event) for event in final_events}
    now = (
        datetime.fromisoformat(str(wall_clock_base_override).replace("Z", "+00:00"))
        if wall_clock_base_override is not None
        else datetime.now(timezone.utc)
    ).astimezone(timezone.utc).replace(microsecond=0)
    observed = now - timedelta(minutes=10)
    if observed_at_override is not None and observed_after is not None:
        raise AssertionError("test archive cannot combine an explicit observation time and lower bound")
    if observed_at_override is not None:
        observed = datetime.fromisoformat(
            str(observed_at_override).replace("Z", "+00:00"),
        ).astimezone(timezone.utc)
    else:
        prior_observations = [
            datetime.fromisoformat(str(item["observed_at"]).replace("Z", "+00:00"))
            for item in raw_archive.load_manifest(raw_root)
            if item.get("observed_at")
        ]
        if prior_observations:
            observed = max(
                observed,
                max(prior_observations).astimezone(timezone.utc) + timedelta(seconds=1),
            )
    if observed_after is not None:
        after = datetime.fromisoformat(str(observed_after).replace("Z", "+00:00"))
        observed = max(observed, after.astimezone(timezone.utc) + timedelta(minutes=1))
    started = observed - timedelta(minutes=1)
    finished = (
        datetime.fromisoformat(str(finished_at_override).replace("Z", "+00:00")).astimezone(timezone.utc)
        if finished_at_override is not None else observed + timedelta(minutes=1)
    )
    if finished < observed:
        raise AssertionError("test archive successful-fetch availability must not precede source observation")
    observed_text = observed.isoformat().replace("+00:00", "Z")
    started_text = started.isoformat().replace("+00:00", "Z")
    finished_text = finished.isoformat().replace("+00:00", "Z")
    future_kickoff_text = (now + timedelta(days=30)).isoformat().replace("+00:00", "Z")
    final_kickoff_text = (observed - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    for event in targets:
        final = event in final_set
        conn.execute(
            "UPDATE events SET finished=?,data_checked=?,updated_at=? WHERE id=?",
            (int(final), int(final), observed_text, event),
        )
        conn.execute(
            "UPDATE fixtures SET kickoff_time=?,started=?,finished=?,finished_provisional=?,updated_at=? "
            "WHERE event=?",
            (final_kickoff_text if final else future_kickoff_text,
             int(final), int(final), int(final), observed_text, event),
        )
    event_payload = {
        "elements": [dict(row) for row in conn.execute(
            "SELECT id,web_name FROM players ORDER BY id",
        )],
        "teams": [dict(row) for row in conn.execute(
            "SELECT id,name FROM teams ORDER BY id",
        )],
        "events": [dict(row) for row in conn.execute(
            "SELECT id,finished,data_checked FROM events WHERE id IN ("
            + ",".join("?" for _ in targets) + ") ORDER BY id",
            targets,
        )],
        "game_settings": {},
    }
    fixture_payload = [dict(row) for row in conn.execute(
        "SELECT id,event,team_h,team_a,kickoff_time,started,finished,finished_provisional "
        "FROM fixtures WHERE event IN (" + ",".join("?" for _ in targets) + ") ORDER BY id",
        targets,
    )]
    live_players = {
        int(event): [int(player_id) for player_id in player_ids]
        for event, player_ids in (live_players_by_event or {}).items()
    }
    endpoints_ok = ["bootstrap-static", "fixtures"] + [
        f"event/{event}/live" for event in sorted(live_players)
    ]
    run_id = repo.create_fetch_run(
        conn, "fetch_fpl", raw_dir=str(raw_root), started_at=started_text,
    )
    repo.finish_fetch_run(
        conn, run_id, "success", current_event=max(targets),
        endpoints_ok=endpoints_ok,
    )
    conn.execute(
        "UPDATE fetch_runs SET started_at=?,finished_at=? WHERE id=?",
        (started_text, finished_text, int(run_id)),
    )
    raw_archive.archive_raw_capture(
        raw_root, source="bootstrap_static", observed_at=observed_text,
        body=json.dumps(event_payload, separators=(",", ":")).encode("utf-8"), run_id=run_id,
    )
    raw_archive.archive_raw_capture(
        raw_root, source="fixtures", observed_at=observed_text,
        body=json.dumps(fixture_payload, separators=(",", ":")).encode("utf-8"), run_id=run_id,
    )
    for event, player_ids in sorted(live_players.items()):
        if event not in targets or event not in final_set:
            raise AssertionError("test event-live archives must be requested only for final target events")
        payload = {
            "elements": [
                {"id": player_id, "stats": {"minutes": 90, "total_points": 1 + player_id % 7}}
                for player_id in player_ids
            ],
        }
        raw_archive.archive_raw_capture(
            raw_root, source=f"event_live_{event}", observed_at=observed_text,
            body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            event=event, run_id=run_id,
        )
    conn.commit()
    return {"run_id": int(run_id), "observed_at": observed_text, "finished_at": finished_text}


def _load_forecast_origin_with_json_overrides(
    monkeypatch, evidence_ops, conn, *, identity, action, observation_id, overrides,
):
    from pathlib import Path

    original_read_json = evidence_ops._read_json
    normalized = {Path(path).resolve(): value for path, value in overrides.items()}

    def read_json(path):
        resolved = Path(path).resolve()
        if resolved in normalized:
            return normalized[resolved]
        return original_read_json(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(evidence_ops, "_read_json", read_json)
        return evidence_ops._load_forecast_origin(
            conn, identity=identity, action=action, observation_id=observation_id,
        )


def _load_chip_evidence_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "chip_evidence.py"
    spec = importlib.util.spec_from_file_location("chip_evidence_horizon_integration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tampered_forecast_loader_artifacts(evidence_ops, evidence_root, operation_id, mutate):
    metadata_root = evidence_ops._metadata_root(evidence_root)
    intent_path = metadata_root / f"forecast-intent-{operation_id}.json"
    publication_path = metadata_root / f"forecast-publication-{operation_id}.json"
    receipt_path = metadata_root / f"forecast-{operation_id}.json"
    intent = evidence_ops._read_json(intent_path)
    publication = evidence_ops._read_json(publication_path)
    receipt = evidence_ops._read_forecast_receipt(evidence_root, operation_id)
    intent_body = dict(intent)
    publication_body = dict(publication)
    receipt_body = dict(receipt)
    intent_body.pop("receipt_sha256", None)
    publication_body.pop("receipt_sha256", None)
    receipt_body.pop("receipt_sha256", None)
    mutate(intent_body, publication_body, receipt_body)
    tampered_intent = evidence_ops._sealed(evidence_ops.FORECAST_INTENT_SCHEMA, intent_body)
    publication_body["intent_sha256"] = evidence_ops._sha256(tampered_intent)
    tampered_publication = evidence_ops._sealed(
        evidence_ops.FORECAST_PUBLICATION_SCHEMA, publication_body,
    )
    receipt_body["publication_sha256"] = evidence_ops._sha256(tampered_publication)
    receipt_body["forecast_available_at"] = publication_body.get("forecast_available_at")
    tampered_receipt = evidence_ops._sealed(evidence_ops.FORECAST_RECEIPT_SCHEMA, receipt_body)
    return {
        intent_path: tampered_intent,
        publication_path: tampered_publication,
        receipt_path: tampered_receipt,
    }


def test_projection_uses_verified_route_transition_inside_saved_horizon():
    route, _ = _route()

    state, model = _project_future_manager_state(route, target_event=5, rules=RULES)

    assert model == "VERIFIED_NORMAL_ROUTE_TRANSITIONS_V1"
    assert state.event == 5
    assert state.free_transfers == 4
    assert state.event_start_free_transfers == 4
    assert tuple(sorted(state.by_id())) == tuple(range(1, 16))
    assert state.bank_tenths == 7


def test_projection_after_saved_horizon_uses_canonical_no_transfer_ft_rollover():
    route, _ = _route()

    state, model = _project_future_manager_state(route, target_event=9, rules=RULES)

    assert model == NO_TRANSFER_CONTINUATION_MODEL
    assert state.event == 9
    assert state.free_transfers == 5
    assert state.event_start_free_transfers == 5
    assert tuple(sorted(state.by_id())) == tuple(range(1, 16))


def test_projection_refuses_unknown_origin_event_start_ft_instead_of_copying_current_ft():
    route, initial = _route(event_start_ft=None)
    assert initial.free_transfers == 2

    with pytest.raises(FutureStateAssemblyError, match="event-start free transfers are unknown"):
        _project_future_manager_state(route, target_event=5, rules=RULES)


def test_projection_refuses_target_not_after_origin():
    route, _ = _route()

    with pytest.raises(FutureStateAssemblyError, match="must follow the verified origin"):
        _project_future_manager_state(route, target_event=3, rules=RULES)


def test_roll_route_uses_canonical_transition_for_four_event_save_schedule():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = tuple(
        RoutePlayer(
            player_id=pid,
            position=position,
            club_id=((pid - 1) // 3) + 1,
            purchase_price_tenths=50,
        )
        for pid, position in enumerate(positions, start=1)
    )
    meta = {
        player.player_id: PlayerMeta(player.player_id, player.position, player.club_id)
        for player in players
    }
    start = RouteState(
        event=10,
        players=players,
        bank_tenths=7,
        free_transfers=3,
        event_start_free_transfers=3,
    )

    route = _build_canonical_roll_route(
        start, events=(10, 11, 12, 13), player_meta=meta, rules=RULES,
    )

    assert [action["event"] for action in route.actions] == [10, 11, 12, 13]
    assert [action["kind"] for action in route.actions] == ["ROLL"] * 4
    assert [action["transition"].hit_points for action in route.actions] == [0] * 4
    assert route.state.event == 14
    assert route.state.free_transfers == 5
    assert route.state.event_start_free_transfers == 5
    assert route.state.bank_tenths == 7
    assert route.state.squad_hash() == start.squad_hash()


def test_future_save_route_preserves_real_transfer_suffix_then_rolls_only_beyond_it():
    source_route, _ = _route(transfer_at_event=5)
    player_meta = {
        player_id: PlayerMeta(player_id, position, source_route.clubs[player_id])
        for player_id, position in source_route.positions.items()
    }

    assembled = _build_future_save_route(
        source_route,
        target_event=5,
        horizon_length=4,
        rules=RULES,
        player_meta=player_meta,
        season_rules_evidence={"source": "fixture-only"},
    )

    assert assembled.retained_route_suffix_events == (5, 6)
    assert assembled.no_transfer_continuation_events == (7, 8)
    assert assembled.continuation_model == NO_TRANSFER_CONTINUATION_MODEL
    assert tuple(action["event"] for action in assembled.partial.actions) == (5, 6, 7, 8)
    # GW5's verified transfer is retained, including canonical changed bank/basis.
    gw5 = assembled.partial.actions[0]
    assert gw5 is source_route.partial.actions[2]
    assert gw5["kind"] == "SINGLE"
    assert gw5["transition"].bank_after_tenths == 2
    assert gw5["transition"].next_event_state.by_id()[16].purchase_price_tenths == 55
    # The appended continuation adds no transfers and preserves that new basis.
    assert [len(action["batch"].actions) for action in assembled.partial.actions] == [1, 0, 0, 0]
    assert assembled.partial.state.by_id()[16].purchase_price_tenths == 55


def test_pinned_ft_cap_must_match_transfer_state_before_projection_or_rollover():
    route, _ = _route()
    incompatible = SeasonRules(season="2026/27", max_extra_free_transfers=3, max_free_transfers=4)

    with pytest.raises(FutureStateAssemblyError, match="max_free_transfers=4.*transfer_state=5"):
        _project_future_manager_state(route, target_event=9, rules=incompatible)


def test_future_route_keeps_verified_lineups_inside_source_route():
    route, _ = _route()
    policy = ml.ManagerPolicy(
        starter_ids=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        bench_gk_id=2,
        bench_outfield_order=(6, 7, 12),
        captain_id=13,
        vice_captain_id=14,
    )
    record = {
        "per_event": [
            {"event": event, "policy": policy.as_dict()}
            for event in route.events
        ],
    }

    assembled = _add_future_lineup_policies(
        route.partial,
        route=route,
        source_route_record=record,
        worlds_by_event={},
        positions=route.positions,
    )

    assert [row["policy"] for row in assembled.actions] == [policy.as_dict()] * 4
    assert [row["transition"] for row in assembled.actions] == [
        row["transition"] for row in route.partial.actions
    ]


def test_future_no_transfer_continuation_uses_ranked_policy_not_missing_route_policy(monkeypatch):
    route, _ = _route()
    policy = ml.ManagerPolicy(
        starter_ids=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        bench_gk_id=2,
        bench_outfield_order=(6, 7, 12),
        captain_id=13,
        vice_captain_id=14,
    )
    calls = []

    def rank(squad_ids, positions, matrix, *, top_k):
        calls.append((tuple(squad_ids), matrix, top_k))
        return {"top_policies": [policy]}

    monkeypatch.setattr(ml, "rank_policies", rank)
    continuation = _build_future_save_route(
        route,
        target_event=9,
        horizon_length=2,
        rules=RULES,
        player_meta={
            pid: PlayerMeta(pid, position, route.clubs[pid])
            for pid, position in route.positions.items()
        },
        season_rules_evidence={"source": "fixture-only"},
    ).partial

    assembled = _add_future_lineup_policies(
        continuation,
        route=route,
        source_route_record={"per_event": []},
        worlds_by_event={9: {"event": 9}, 10: {"event": 10}},
        positions=route.positions,
    )

    assert [row["event"] for row in assembled.actions] == [9, 10]
    assert [row["policy"] for row in assembled.actions] == [policy.as_dict()] * 2
    assert [(row[1]["event"], row[2]) for row in calls] == [(9, 1), (10, 1)]


def test_origin_loader_requires_explicit_decision_generation_route_and_manager_identity():
    with pytest.raises(FutureStateAssemblyError, match="explicit origin identity is incomplete"):
        load_projected_future_manager_state(
            None,
            decision_id="",
            generation_id="",
            route_id="",
            entry_id=0,
            origin_cutoff="",
            data_snapshot_sha256="",
            target_event=10,
            season="",
        )


def test_public_origin_loader_reverifies_real_tiny_decision_route_rules_and_transfer_suffix(
    tmp_path, monkeypatch,
):
    from fpl_brain import chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_route_assembly as cra

    conn, generation, decision_id, route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=2,
    )
    try:
        route = cra.load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        projected = assembly.load_projected_future_manager_state(
            conn,
            decision_id=decision_id,
            generation_id=generation.generation_id,
            route_id=route_id,
            entry_id=241392,
            origin_cutoff=generation.cutoff,
            data_snapshot_sha256=str(generation.snapshot["sha256"]),
            target_event=5,
            season="2026/27",
        )
        assert projected.state.event == 5
        assert projected.state.event_start_free_transfers == 3
        assert projected.season_rules.max_free_transfers == ts.MAX_STORED_FREE_TRANSFERS
        assert projected.season_rules_evidence["source"] == (
            "ACCEPTED_BOOTSTRAP_CAPTURE_IN_ORIGIN_SNAPSHOT_ARCHIVE"
        )
        assert projected.as_dict()["classification"] == "PROJECTED_ASSUMED_NOT_OFFICIAL_MANAGER_FACT"

        player_meta = {
            pid: PlayerMeta(pid, position, route.clubs[pid])
            for pid, position in route.positions.items()
        }
        future_save = assembly._build_future_save_route(
            route,
            target_event=5,
            horizon_length=4,
            rules=projected.season_rules,
            player_meta=player_meta,
            season_rules_evidence=projected.season_rules_evidence,
        )
        assert future_save.retained_route_suffix_events == (5, 6, 7)
        assert future_save.no_transfer_continuation_events == (8,)
        gw5 = future_save.partial.actions[0]
        assert gw5["kind"] == "SINGLE"
        assert gw5["transition"].bank_after_tenths == 4
        assert gw5["transition"].next_event_state.by_id()[43].purchase_price_tenths == 50
        assert [len(action["batch"].actions) for action in future_save.partial.actions] == [1, 0, 0, 0]

        with pytest.raises(FutureStateAssemblyError, match="differs from the explicit origin identity"):
            assembly.load_projected_future_manager_state(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241392,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256="sha256:" + "f" * 64,
                target_event=5,
                season="2026/27",
            )
        with pytest.raises(FutureStateAssemblyError, match="manager entry differs"):
            assembly.load_projected_future_manager_state(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241393,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256=str(generation.snapshot["sha256"]),
                target_event=5,
                season="2026/27",
            )
    finally:
        conn.close()


def test_public_origin_loader_keeps_unknown_event_start_ft_blocked_in_real_route(
    tmp_path, monkeypatch,
):
    from fpl_brain import chip_horizon_evidence_assembly as assembly

    conn, generation, decision_id, route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=None,
    )
    try:
        with pytest.raises(FutureStateAssemblyError, match="event-start free transfers are unknown"):
            assembly.load_projected_future_manager_state(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241392,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256=str(generation.snapshot["sha256"]),
                target_event=5,
                season="2026/27",
            )
    finally:
        conn.close()


def test_future_composers_do_not_expose_world_materialization_override():
    import inspect

    from fpl_brain.chip_horizon_evidence_assembly import (
        build_future_free_hit_request,
        build_future_wildcard_request,
    )

    assert "allow_materialization" not in inspect.signature(build_future_free_hit_request).parameters
    assert "allow_materialization" not in inspect.signature(build_future_wildcard_request).parameters


def test_public_future_composer_refuses_unknown_ft_and_snapshot_tampering(tmp_path, monkeypatch):
    from fpl_brain import chip_horizon_evidence_assembly as assembly

    conn, generation, decision_id, route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=None, include_event8=True,
    )
    try:
        with pytest.raises(FutureStateAssemblyError, match="event-start free transfers are unknown"):
            assembly.build_future_free_hit_request(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241392,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256=str(generation.snapshot["sha256"]),
                target_event=5,
                season="2026/27",
                continuation_generation_id=generation.generation_id,
                cache_dir=tmp_path / "unknown-ft-cache",
                expiry_event=5,
                coverage_product={},
            )
    finally:
        conn.close()


def test_public_future_fh_composer_refuses_expired_pinned_chip_window(tmp_path, monkeypatch):
    import generation_fixtures as gf
    from datetime import datetime, timedelta, timezone
    from fpl_brain import chip_decision as cd, chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_reservation_forecast as crf, chip_route_assembly as cra
    from fpl_brain import execution, generation_store as gs, monte_carlo

    conn, generation, decision_id, route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=2, extended_events=(8, 9),
    )
    cache_dir = tmp_path / "expired-chip-fixture-cache"
    try:
        route = cra.load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        runs = {event: generation.runs_for(event) for event in generation.events}
        for event in (8, 9):
            runs[event] = {
                str(row["model_family"]): int(row["id"])
                for row in conn.execute(
                    "SELECT id,model_family FROM projection_runs WHERE planning_event=?", (event,)
                )
            }
        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=4,
            planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="public_expired_fh_window_fixture",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            continuation = gs.certify_chip_reservation_generation(
                conn,
                chip_generation_id=generation.generation_id,
                planning_event=4,
                cutoff=generation.cutoff,
                events=(4, 5, 6, 7, 8, 9),
                runs_by_event=runs,
                snapshot=gf.fixture_snapshot(conn),
                controller=controller,
            )
        except BaseException as failure:
            controller.finish(execution.RUN_FAILED, str(failure))
            raise
        controller.finish(execution.RUN_COMPLETE)
        coverage = crf.build_reservation_coverage_product(
            conn,
            action=cd.CHIP_ACTION_FH,
            source_identity=assembly._source_identity(route, generation),
            expiry_event=6,
            product_generation_id=continuation.generation_id,
        )
        _materialize_synthetic_fixture_caches(
            conn,
            continuation,
            decision_id,
            cache_dir,
            events=(6, 7, 8, 9),
            windows=((6, 7, 8, 9),),
        )

        def forbid_world_simulation(*_args, **_kwargs):
            raise AssertionError("expired-chip public composer attempted to materialize worlds")

        monkeypatch.setattr(monte_carlo, "simulate", forbid_world_simulation)
        with pytest.raises(FutureStateAssemblyError, match="does not establish FH eligibility at GW6"):
            assembly.build_future_free_hit_request(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241392,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256=str(generation.snapshot["sha256"]),
                target_event=6,
                season="2026/27",
                continuation_generation_id=continuation.generation_id,
                cache_dir=cache_dir,
                expiry_event=6,
                coverage_product=coverage,
            )
    finally:
        conn.close()

    tamper_root = tmp_path / "snapshot-tamper"
    tamper_root.mkdir()
    conn, generation, decision_id, route_id = _make_verified_origin(
        tamper_root, monkeypatch, event_start_ft=2,
    )
    try:
        with pytest.raises(FutureStateAssemblyError, match="differs from the explicit origin identity"):
            assembly.build_future_wildcard_request(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241392,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256="f" * 64,
                target_event=5,
                season="2026/27",
                continuation_generation_id=generation.generation_id,
                value_generation_id=generation.generation_id,
                cache_dir=tmp_path / "snapshot-tamper-cache",
                expiry_event=5,
                coverage_product={},
            )
    finally:
        conn.close()


def test_pinned_chip_window_is_selected_and_expiry_must_match():
    from fpl_brain.chip_decision import CHIP_ACTION_FH, CHIP_ACTION_TO_OFFICIAL_NAME
    from fpl_brain.chip_horizon_evidence_assembly import _verify_chip_window

    name = CHIP_ACTION_TO_OFFICIAL_NAME[CHIP_ACTION_FH]
    rows = [
        {
            "name": name, "window_start_event": 2, "window_stop_event": 4,
            "available_for_event": False, "used": True, "expired": True,
        },
        {
            "name": name, "window_start_event": 5, "window_stop_event": 19,
            "available_for_event": True, "used": False, "expired": False,
        },
    ]
    selected = _verify_chip_window(rows, action=CHIP_ACTION_FH, event=7, expiry_event=19)
    assert selected["eligible"] is True
    assert selected["window_stop_event"] == 19
    with pytest.raises(FutureStateAssemblyError, match="does not bind its declared expiry"):
        _verify_chip_window(rows, action=CHIP_ACTION_FH, event=7, expiry_event=38)


def test_future_coverage_must_bind_the_explicit_verified_continuation_generation(tmp_path):
    """A valid coverage product for another generation is not interchangeable."""
    import generation_fixtures as gf
    import test_pe9_certification_integration as pe9
    from fpl_brain import chip_decision as cd, chip_reservation_forecast as crf
    from fpl_brain import chip_horizon_evidence_assembly as assembly
    from fpl_brain import execution, generation_store as gs
    from datetime import datetime, timedelta, timezone

    events = tuple(range(1, 9))
    conn, runs = pe9._world(events=events)
    try:
        snapshot = gf.fixture_snapshot(conn)
        normal_events = events[:4]
        normal = gf.certify_world(
            conn,
            {event: runs[event] for event in normal_events},
            events=normal_events,
            planning_event=1,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            snapshot_path=tmp_path / "normal-origin.db",
        )

        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=1,
            planning_cutoff=normal.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="future_coverage_generation_binding_test",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            continuation = gs.certify_chip_reservation_generation(
                conn,
                chip_generation_id=normal.generation_id,
                planning_event=1,
                cutoff=normal.cutoff,
                events=events,
                runs_by_event=runs,
                snapshot=snapshot,
                controller=controller,
            )
        except BaseException as failure:
            controller.finish(execution.RUN_FAILED, str(failure))
            raise
        controller.finish(execution.RUN_COMPLETE)

        per_event = normal.manifest["per_event"]
        certification_identity = gs._certification_identity_for_bundles(
            cutoff=str(normal.cutoff),
            bundle_identities={
                str(event): str(per_event[str(event)]["bundle_identity"])
                for event in normal.events
            },
            snapshot_sha256=str(normal.snapshot["sha256"]),
        )
        route = SimpleNamespace(
            source_decision_id="fixture-decision",
            source_result_sha256="sha256:" + "1" * 64,
            source_artifact_sha256="2" * 64,
            generation_id=normal.generation_id,
            planning_event=1,
            cutoff=normal.cutoff,
            data_snapshot_sha256=normal.snapshot["sha256"],
            certification_identity=certification_identity,
        )
        source_identity = assembly._source_identity(route, normal)
        product = crf.build_reservation_coverage_product(
            conn,
            action=cd.CHIP_ACTION_FH,
            source_identity=source_identity,
            expiry_event=5,
            product_generation_id=continuation.generation_id,
        )

        verified = assembly._verify_future_coverage(
            conn,
            route=route,
            generation=normal,
            action=cd.CHIP_ACTION_FH,
            event=5,
            expiry_event=5,
            continuation_generation_id=continuation.generation_id,
            coverage_product=product,
        )
        assert verified["product_generation_id"] == continuation.generation_id

        with pytest.raises(FutureStateAssemblyError, match="does not authorize GW5 through expiry"):
            assembly._verify_future_coverage(
                conn,
                route=route,
                generation=normal,
                action=cd.CHIP_ACTION_FH,
                event=5,
                expiry_event=5,
                continuation_generation_id="different-explicit-generation",
                coverage_product=product,
            )
    finally:
        conn.close()


def test_minutes_loader_reads_the_frozen_minutes_v1_family():
    import test_pe9_certification_integration as pe9
    from fpl_brain import analytics
    from fpl_brain.chip_horizon_evidence_assembly import _load_minutes_projection_rows

    conn, runs = pe9._world(events=(1,))
    try:
        run_id = int(runs[1]["minutes_v1"])
        analytics.freeze_prediction(
            conn,
            run_id,
            kind=analytics.MINUTES_V1_KIND,
            player_id=1,
            event=1,
            fixture_id=1000,
            payload={"expected_minutes": 71.5},
            model_version="fixture-minutes-v1",
        )
        assert _load_minutes_projection_rows(conn, run_id) == {
            (1, 1000): {"expected_minutes": 71.5},
        }
    finally:
        conn.close()


def test_public_future_fh_composer_requires_cache_then_reconstructs_routes(
    tmp_path, monkeypatch,
):
    import generation_fixtures as gf
    from datetime import datetime, timedelta, timezone
    from fpl_brain import candidate_universe as cu
    from fpl_brain import chip_decision as cd, chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_reservation_forecast as crf, chip_route_assembly as cra
    from fpl_brain import execution, generation_store as gs, monte_carlo

    conn, generation, decision_id, route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=2, include_event8=True,
    )
    cache_dir = tmp_path / "must-stay-absent-world-cache"
    try:
        route = cra.load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        runs = {event: generation.runs_for(event) for event in generation.events}
        runs[8] = {
            str(row["model_family"]): int(row["id"])
            for row in conn.execute(
                "SELECT id,model_family FROM projection_runs WHERE planning_event=8"
            )
        }
        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=4,
            planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="public_future_fh_cache_only_fixture",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            continuation = gs.certify_chip_reservation_generation(
                conn,
                chip_generation_id=generation.generation_id,
                planning_event=4,
                cutoff=generation.cutoff,
                events=(4, 5, 6, 7, 8),
                runs_by_event=runs,
                snapshot=gf.fixture_snapshot(conn),
                controller=controller,
            )
        except BaseException as failure:
            controller.finish(execution.RUN_FAILED, str(failure))
            raise
        controller.finish(execution.RUN_COMPLETE)

        coverage = crf.build_reservation_coverage_product(
            conn,
            action=cd.CHIP_ACTION_FH,
            source_identity=assembly._source_identity(route, generation),
            expiry_event=5,
            product_generation_id=continuation.generation_id,
        )
        def forbid_world_simulation(*_args, **_kwargs):
            raise AssertionError("public FH composition attempted to materialize worlds")

        original_simulate = monte_carlo.simulate
        monkeypatch.setattr(monte_carlo, "simulate", forbid_world_simulation)
        with pytest.raises(FutureStateAssemblyError, match="certified continuation worlds for GW5 are unavailable"):
            assembly.build_future_free_hit_request(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241392,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256=str(generation.snapshot["sha256"]),
                target_event=5,
                season="2026/27",
                continuation_generation_id=continuation.generation_id,
                cache_dir=cache_dir,
                expiry_event=5,
                coverage_product=coverage,
            )
        assert not cache_dir.exists()

        # Build the test-only cache from the certified tiny fixture using the
        # accepted loader, then restore the spy for both composer success paths.
        monkeypatch.setattr(monte_carlo, "simulate", original_simulate)
        _materialize_synthetic_fixture_caches(
            conn,
            continuation,
            decision_id,
            cache_dir,
            events=(5, 6, 7, 8),
            windows=((5, 6, 7, 8), (6, 7, 8)),
        )
        monkeypatch.setattr(monte_carlo, "simulate", forbid_world_simulation)
        request, evidence = assembly.build_future_free_hit_request(
            conn,
            decision_id=decision_id,
            generation_id=generation.generation_id,
            route_id=route_id,
            entry_id=241392,
            origin_cutoff=generation.cutoff,
            data_snapshot_sha256=str(generation.snapshot["sha256"]),
            target_event=5,
            season="2026/27",
            continuation_generation_id=continuation.generation_id,
            cache_dir=cache_dir,
            expiry_event=5,
            coverage_product=coverage,
        )
        assert tuple(row.event for row in request.save_route.events) == (5, 6, 7, 8)
        assert tuple(row.event for row in request.play_route.events) == (6, 7, 8)
        assert request.play_route.start_state["squad_ids"] == list(request.permanent.owned_ids)
        assert request.permanent.bank_tenths == 7
        assert request.permanent.free_transfers == 3
        assert request.play_route.start_state["bank_tenths"] == 7
        assert request.play_route.start_state["free_transfers"] == 3
        gw5 = request.save_route.event_for(5)
        assert 43 in gw5.squad_ids and 13 not in gw5.squad_ids
        assert gw5.bank_tenths == 4
        assert gw5.free_transfers == 3
        assert gw5.purchase_price_tenths[43] == 50
        assert 13 not in gw5.purchase_price_tenths
        gw5_action = next(row for row in route.partial.actions if int(row["event"]) == 5)
        assert [(move.out_player_id, move.in_player_id) for move in gw5_action["batch"].actions] == [(13, 43)]
        retained_policy = cra._policy_from_route(assembly._route_record(route), event=5)
        assert 43 in retained_policy.starter_ids
        # The DTO policy is the accepted exact evaluator's fresh rank_policies
        # choice, not the attached route-policy mapping above.
        assert dict(request.save_route.event_for(5).policy) != retained_policy.as_dict()
        assert evidence["coverage_product_sha256"] == coverage["product_sha256"]
        assert all(row["status"] == "HIT" for row in evidence["world_cache_report"])

        _archive_future_official_state(conn, tmp_path / "origin-raw", [5, 6, 7, 8])
        from fpl_brain import chip_horizon_evidence_ops as horizon_ops
        from fpl_brain import chip_evidence_ops as evidence_ops

        evidence_root = tmp_path / "fh-operator-evidence"
        identity = evidence_ops._required_identity(
            db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            evidence_root=evidence_root,
        )
        origin = horizon_ops.register_horizon_origin(
            conn,
            db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            action="FH", evidence_root=evidence_root,
        )
        assert origin["origin"]["chip_eligibility"]["expiry_event"] == 5
        operation_id = evidence_ops._forecast_operation_id(
            evidence_ops._sha256(evidence_ops._origin_key(identity)), "FH", "fh-source-forecast-1",
        )
        _archive_future_official_state(
            conn, tmp_path / "origin-raw", [5, 6, 7, 8], final_events=[8],
        )
        with pytest.raises(evidence_ops.ChipEvidenceError, match="GW8 is FINAL"):
            horizon_ops.build_future_action_forecast(
                conn, identity=identity, action="FH", observation_id="fh-source-forecast-1",
                entry_id=241392, continuation_generation_id=continuation.generation_id,
                value_generation_ids={}, expiry_event=5, coverage_product=coverage,
                evidence_root=evidence_root, cache_dir=cache_dir,
            )
        metadata_root = evidence_ops._metadata_root(evidence_root)
        assert not (metadata_root / f"forecast-intent-{operation_id}.json").exists()
        assert not (metadata_root / f"forecast-publication-{operation_id}.json").exists()
        assert not (metadata_root / f"forecast-{operation_id}.json").exists()

        _archive_future_official_state(conn, tmp_path / "origin-raw", [5, 6, 7, 8])
        hit_gate_calls = []
        original_hit_gate = assembly.verify_future_action_hit_cost_witnesses
        def audited_fh_gate(conn, **kwargs):
            assert tuple(kwargs["expected_events"]) == (5,)
            assert tuple(int(row["event"]) for row in kwargs["opportunity_records"]) == (5,)
            proof = original_hit_gate(conn, **kwargs)
            hit_gate_calls.append((5,))
            monkeypatch.setattr(
                assembly, "verify_future_action_hit_cost_witnesses", original_hit_gate,
            )
            return proof
        monkeypatch.setattr(assembly, "verify_future_action_hit_cost_witnesses", audited_fh_gate)
        original_retain = crf.retain_event_opportunity_record
        def fh_retain_only_after_source_gate(*args, **kwargs):
            assert hit_gate_calls == [(5,)]
            return original_retain(*args, **kwargs)
        monkeypatch.setattr(crf, "retain_event_opportunity_record", fh_retain_only_after_source_gate)
        forecast = horizon_ops.build_future_action_forecast(
            conn, identity=identity, action="FH", observation_id="fh-source-forecast-1",
            entry_id=241392, continuation_generation_id=continuation.generation_id,
            value_generation_ids={}, expiry_event=5, coverage_product=coverage,
            evidence_root=evidence_root, cache_dir=cache_dir,
        )
        assert hit_gate_calls == [(5,)]
        assert forecast["forecast"]["coverage_complete"] is True
        assert forecast["forecast"]["covered_events"] == [5]
        assert forecast["receipt"]["selected_event"] == 5
        assert forecast["receipt"]["action"] == cd.CHIP_ACTION_FH
        assert forecast["receipt"]["materialized_worlds"] is False
        assert forecast["receipt"]["valuation_state_events"] == [5, 6, 7, 8]
        assert evidence_ops._parse_utc(
            forecast["receipt"]["forecast_available_at"], name="test forecast_available_at",
        ) >= evidence_ops._parse_utc(forecast["receipt"]["issued_at"], name="test issued_at")
        assert forecast["cache_report"] and all(row["status"] == "HIT" for row in forecast["cache_report"])
        assert all(row["artifact_payload"].get("outcome_arms") for row in forecast["forecast"]["opportunities"])
        publication_path = metadata_root / f"forecast-publication-{operation_id}.json"
        publication = evidence_ops._read_json(publication_path)
        assert [row["event"] for row in publication["valuation_prospective_availability"]] == [5, 6, 7, 8]
        intent = evidence_ops._read_json(metadata_root / f"forecast-intent-{operation_id}.json")

        common = evidence_ops._load_forecast_origin(
            conn, identity=identity, action="FH", observation_id="fh-source-forecast-1",
        )
        assert common[5]["receipt"] == forecast["receipt"]
        assert common[5]["forecast"] == forecast["forecast"]

        def assert_common_loader_rejects(mutator, match):
            overrides = _tampered_forecast_loader_artifacts(
                evidence_ops, evidence_root, operation_id, mutator,
            )
            with pytest.raises(evidence_ops.ChipEvidenceError, match=match):
                _load_forecast_origin_with_json_overrides(
                    monkeypatch, evidence_ops, conn, identity=identity, action="FH",
                    observation_id="fh-source-forecast-1", overrides=overrides,
                )

        assert_common_loader_rejects(
            lambda _intent, publication_body, _receipt: publication_body.update(
                forecast_available_at="",
            ),
            "forecast_available_at",
        )
        assert_common_loader_rejects(
            lambda _intent, publication_body, _receipt: publication_body.update(
                forecast_available_at=publication_body["issued_at"],
            ),
            "final prospective state check",
        )
        assert_common_loader_rejects(
            lambda _intent, publication_body, _receipt: publication_body.update(
                forecast_available_at="2099-01-01T00:00:00Z",
            ),
            "target fixture kickoff",
        )

        def remove_late_valuation_event(intent_body, publication_body, receipt_body):
            intent_body["valuation_state_events"] = intent_body["valuation_state_events"][:-1]
            intent_body["valuation_preflight"] = intent_body["valuation_preflight"][:-1]
            intent_body["valuation_calculation_start"] = intent_body["valuation_calculation_start"][:-1]
            publication_body["valuation_state_events"] = publication_body["valuation_state_events"][:-1]
            publication_body["valuation_prospective_issuance"] = publication_body[
                "valuation_prospective_issuance"
            ][:-1]
            publication_body["valuation_prospective_availability"] = publication_body[
                "valuation_prospective_availability"
            ][:-1]
            receipt_body["valuation_state_events"] = receipt_body["valuation_state_events"][:-1]

        assert_common_loader_rejects(
            remove_late_valuation_event,
            "valuation-state event set differs from certified action horizons",
        )
        assert_common_loader_rejects(
            lambda _intent, publication_body, _receipt: publication_body.update(
                valuation_prospective_availability=publication_body[
                    "valuation_prospective_availability"
                ][:-1],
            ),
            "does not cover every target event",
        )
        def remove_zero_hit_proof(_intent, publication_body, receipt_body):
            publication_body.pop("hit_cost_proof_sha256", None)
            for report in publication_body.get("assembly_reports") or ():
                report.pop("hit_cost_witness", None)
            receipt_body.pop("hit_cost_proof_sha256", None)

        assert_common_loader_rejects(
            remove_zero_hit_proof,
            "source-replayed zero-hit proof refused",
        )

        def alter_selected_play_route(_intent, publication_body, _receipt):
            witness = publication_body["assembly_reports"][0]["hit_cost_witness"]
            record = witness["play_route_record"]
            record["actions"][0]["event"] = int(record["actions"][0]["event"]) + 1
            witness["play_route_record_sha256"] = crf.canonical_sha256(record)
            witness_body = dict(witness)
            witness_body.pop("witness_sha256", None)
            witness["witness_sha256"] = crf.canonical_sha256(witness_body)

        def malformed_hit_cost_witness(_intent, publication_body, _receipt):
            publication_body["assembly_reports"][0]["hit_cost_witness"] = []

        assert_common_loader_rejects(
            alter_selected_play_route,
            "selected optimizer family identity",
        )

        with pytest.raises(evidence_ops.ChipEvidenceError, match="cannot read retained JSON"):
            evidence_ops._load_forecast_origin(
                conn, identity=identity, action="WC", observation_id="fh-source-forecast-1",
            )
        horizon_receipt = evidence_ops._read_json(
            metadata_root / f"horizon-origin-{origin['origin']['origin_id']}-FH.json",
        )
        horizon_artifact_path = evidence_root / horizon_receipt["artifact_ref"]
        tampered_horizon_artifact = evidence_ops._read_json(horizon_artifact_path)
        tampered_horizon_artifact["action"] = "WC"
        with pytest.raises(evidence_ops.ChipEvidenceError, match="binding does not verify"):
            _load_forecast_origin_with_json_overrides(
                monkeypatch, evidence_ops, conn, identity=identity, action="FH",
                observation_id="fh-source-forecast-1",
                overrides={horizon_artifact_path: tampered_horizon_artifact},
            )

        publication_body = dict(publication)
        publication_body.pop("receipt_sha256", None)
        invalid_publication = evidence_ops._sealed(
            evidence_ops.FORECAST_PUBLICATION_SCHEMA,
            {**publication_body, "forecast_available_at": ""},
        )
        with pytest.raises(evidence_ops.ChipEvidenceError, match="forecast_available_at"):
            horizon_ops._forecast_receipt_from_publication(
                conn, identity=identity, origin_id=origin["origin"]["origin_id"],
                operation_id=operation_id, observation_id="fh-source-forecast-1", action="FH",
                expiry_event=5, publication=invalid_publication, intent=intent,
            )
        replay = horizon_ops.build_future_action_forecast(
            conn, identity=identity, action="FH", observation_id="fh-source-forecast-1",
            entry_id=241392, continuation_generation_id=continuation.generation_id,
            value_generation_ids={}, expiry_event=5, coverage_product=coverage,
            evidence_root=evidence_root, cache_dir=cache_dir,
        )
        assert replay["reused"] is True
        assert replay["receipt"] == forecast["receipt"]
        assert replay["receipt"]["forecast_available_at"] == forecast["receipt"]["forecast_available_at"]
        # Exercise receipt-loss recovery from the complete immutable publication.
        # The same operation must reconstruct only its receipt, not rescore.
        forecast_receipt_path = metadata_root / f"forecast-{operation_id}.json"
        retained_forecast_receipt = forecast_receipt_path.read_bytes()
        forecast_receipt_path.unlink()
        recovered_publication = horizon_ops.build_future_action_forecast(
            conn, identity=identity, action="FH", observation_id="fh-source-forecast-1",
            entry_id=241392, continuation_generation_id=continuation.generation_id,
            value_generation_ids={}, expiry_event=5, coverage_product=coverage,
            evidence_root=evidence_root, cache_dir=cache_dir,
        )
        assert recovered_publication["recovered_publication"] is True
        assert forecast_receipt_path.read_bytes() == retained_forecast_receipt

        parent = evidence_ops._load_forecast_origin(
            conn, identity=identity, action="FH", observation_id="fh-source-forecast-1",
        )[5]
        players_by_event = evidence_ops._schedule_player_map(parent["causal"], "FH")
        schedule_events = list(players_by_event)
        assert schedule_events[0] == 5 and len(schedule_events) >= 2
        late_event = schedule_events[-1]
        omitted_player = players_by_event[late_event][-1]
        capture_root = tmp_path / "fh-outcome-raw"
        early_run = _archive_future_official_state(
            conn, capture_root, schedule_events, final_events=schedule_events,
            live_players_by_event=players_by_event,
        )
        early_capture_kwargs = {
            "db_path": tmp_path / "origin.db", "entry_id": 241392,
            "generation_id": generation.generation_id, "decision_id": decision_id,
            "route_id": route_id, "planning_event": 4, "cutoff": generation.cutoff,
            "action": "FH", "observation_id": "fh-source-forecast-1",
            "fetch_run_id": early_run["run_id"], "raw_root": capture_root,
            "evidence_root": evidence_root,
        }
        with pytest.raises(evidence_ops.ChipEvidenceError, match="not strictly later than the retained forecast availability"):
            evidence_ops.capture_outcome(
                conn, **early_capture_kwargs, realization_event=schedule_events[0],
            )
        assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 0

        forecast_available = evidence_ops._parse_utc(
            forecast["receipt"]["forecast_available_at"], name="test forecast_available_at",
        )
        metadata_root = evidence_ops._metadata_root(evidence_root)

        def capture_artifact_bytes():
            return {
                path.name: path.read_bytes()
                for path in metadata_root.glob("capture-*.json")
            }

        capture_artifacts_before_boundary_checks = capture_artifact_bytes()
        finality_equal_run = _archive_future_official_state(
            conn, capture_root, schedule_events, final_events=schedule_events,
            live_players_by_event=players_by_event,
            observed_at_override=forecast_available.isoformat().replace("+00:00", "Z"),
            finished_at_override=(forecast_available + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        )
        finality_equal_archive = evidence_ops._official_event_archive(
            conn, raw_root=capture_root, fetch_run_id=finality_equal_run["run_id"],
            event=schedule_events[0],
        )
        assert finality_equal_archive["official_final_at"] == forecast["receipt"]["forecast_available_at"]
        assert evidence_ops._parse_utc(
            finality_equal_archive["available_at"], name="test capture availability",
        ) > forecast_available
        with pytest.raises(
            evidence_ops.ChipEvidenceError,
            match="official finality is not strictly later than the retained forecast availability",
        ):
            evidence_ops.capture_outcome(
                conn,
                **{**early_capture_kwargs, "fetch_run_id": finality_equal_run["run_id"]},
                realization_event=schedule_events[0],
            )

        availability_equal_run = _archive_future_official_state(
            conn, capture_root, schedule_events, final_events=schedule_events,
            live_players_by_event=players_by_event,
            observed_at_override=(forecast_available - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            finished_at_override=forecast_available.isoformat().replace("+00:00", "Z"),
        )
        availability_equal_archive = evidence_ops._official_event_archive(
            conn, raw_root=capture_root, fetch_run_id=availability_equal_run["run_id"],
            event=schedule_events[0],
        )
        assert evidence_ops._parse_utc(
            availability_equal_archive["official_final_at"], name="test official finality",
        ) < forecast_available
        assert availability_equal_archive["available_at"] == forecast["receipt"]["forecast_available_at"]
        # With a valid archive, official_final_at <= available_at. Thus the
        # public path must reject this availability-equality capture at the
        # earlier finality check; a standalone availability-only equality
        # cannot be constructed from valid retained source chronology.
        with pytest.raises(
            evidence_ops.ChipEvidenceError,
            match="official finality is not strictly later than the retained forecast availability",
        ):
            evidence_ops.capture_outcome(
                conn,
                **{**early_capture_kwargs, "fetch_run_id": availability_equal_run["run_id"]},
                realization_event=schedule_events[0],
            )
        assert capture_artifact_bytes() == capture_artifacts_before_boundary_checks
        assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 0

        incomplete_players = {
            event: (players if event != late_event else [pid for pid in players if pid != omitted_player])
            for event, players in players_by_event.items()
        }
        incomplete_run = _archive_future_official_state(
            conn, capture_root, schedule_events, final_events=schedule_events,
            live_players_by_event=incomplete_players,
            observed_after=forecast["receipt"]["forecast_available_at"],
        )
        capture_kwargs = {
            "db_path": tmp_path / "origin.db", "entry_id": 241392,
            "generation_id": generation.generation_id, "decision_id": decision_id,
            "route_id": route_id, "planning_event": 4, "cutoff": generation.cutoff,
            "action": "FH", "observation_id": "fh-source-forecast-1",
            "fetch_run_id": incomplete_run["run_id"], "raw_root": capture_root,
            "evidence_root": evidence_root,
        }
        for event in schedule_events[:-1]:
            evidence_ops.capture_outcome(
                conn, **capture_kwargs, realization_event=event,
            )

        # A syntactically plausible FINAL row without a receipt must not let the
        # canonical finalizer select it for the missing late-event player.
        from fpl_brain import outcome_ledger as ol
        ol.capture_observation(
            conn,
            grain=ol.GRAIN_PLAYER_EVENT,
            event=late_event,
            player_id=omitted_player,
            fields={"minutes": 90, "total_points": 6},
            source_name="player_gameweeks_final",
            official_final_at=incomplete_run["observed_at"],
            observation_state=ol.OBSERVATION_FINAL,
            source_identity=f"player_gameweeks:{late_event}",
            source_payload_sha256="f" * 64,
            fetch_run_id=incomplete_run["run_id"],
            captured_at=incomplete_run["finished_at"],
        )
        conn.commit()
        from fpl_brain import chip_reservation_calibration as cal
        original_finalizer = cal.finalize_causal_observation
        finalizer_calls = []

        def spy_finalizer(*args, **kwargs):
            finalizer_calls.append((args, kwargs))
            return original_finalizer(*args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(cal, "finalize_causal_observation", spy_finalizer)
            with pytest.raises(evidence_ops.ChipEvidenceError, match="no verified official outcome-capture receipt"):
                evidence_ops.mature_observation(
                    conn,
                    db_path=tmp_path / "origin.db", entry_id=241392,
                    generation_id=generation.generation_id, decision_id=decision_id,
                    route_id=route_id, planning_event=4, cutoff=generation.cutoff,
                    action="FH", observation_id="fh-source-forecast-1",
                    realization_event=schedule_events[0], evidence_root=evidence_root,
                )
        assert not finalizer_calls
        # Now retain an event receipt whose official payload omits that player.
        # The plausible ledger row is still not receipt-bound and remains a hard
        # stop, even though there is at least one exact event receipt.
        evidence_ops.capture_outcome(
            conn, **capture_kwargs, realization_event=late_event,
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(cal, "finalize_causal_observation", spy_finalizer)
            with pytest.raises(evidence_ops.ChipEvidenceError, match="not uniquely bound"):
                evidence_ops.mature_observation(
                    conn,
                    db_path=tmp_path / "origin.db", entry_id=241392,
                    generation_id=generation.generation_id, decision_id=decision_id,
                    route_id=route_id, planning_event=4, cutoff=generation.cutoff,
                    action="FH", observation_id="fh-source-forecast-1",
                    realization_event=schedule_events[0], evidence_root=evidence_root,
                )
        assert not finalizer_calls
        assert not list(evidence_root.glob("chip-outcome-*.json"))
        assert not list(evidence_ops._metadata_root(evidence_root).glob("maturation-*.json"))

        complete_run = _archive_future_official_state(
            conn, capture_root, schedule_events, final_events=schedule_events,
            live_players_by_event=players_by_event,
            observed_after=incomplete_run["finished_at"],
        )
        for event in schedule_events:
            evidence_ops.capture_outcome(
                conn, **{**capture_kwargs, "fetch_run_id": complete_run["run_id"]},
                realization_event=event,
            )

        # The current ledger-selected FINAL row is newer than an older row
        # already bound to the complete_run receipt, but this append has no
        # capture-operation receipt of its own. It must not be hidden by that
        # older valid evidence or reach the canonical finalizer.
        superseded_player = players_by_event[late_event][0]
        older_bound_manifest = evidence_ops._verified_horizon_capture_manifest(
            conn, evidence_root=evidence_root, identity=identity, parent=parent,
            observation_id="fh-source-forecast-1", action="FH",
        )
        complete_event_receipt = next(
            receipt for receipt in older_bound_manifest[0]
            if receipt["realization_event"] == late_event
            and receipt["fetch_run_id"] == complete_run["run_id"]
        )
        complete_event_selection = next(
            item for item in older_bound_manifest[2]
            if item["event"] == late_event and item["player_id"] == superseded_player
        )
        assert complete_event_selection["operation_id"] == complete_event_receipt["operation_id"]
        newer_unbound_run = _archive_future_official_state(
            conn, capture_root, [late_event], final_events=[late_event],
            live_players_by_event={late_event: players_by_event[late_event]},
            observed_after=complete_run["finished_at"],
        )
        newer_unbound_archive = evidence_ops._official_event_archive(
            conn, raw_root=capture_root, fetch_run_id=newer_unbound_run["run_id"],
            event=late_event,
        )
        newer_live_record = newer_unbound_archive["records"]["event_live"]
        newer_live_player = next(
            row for row in newer_unbound_archive["payloads"]["event_live"]["elements"]
            if int(row["id"]) == superseded_player
        )
        ol.capture_observation(
            conn,
            grain=ol.GRAIN_PLAYER_EVENT,
            event=late_event,
            player_id=superseded_player,
            fields=newer_live_player["stats"],
            source_name="player_gameweeks_final",
            official_final_at=newer_unbound_archive["official_final_at"],
            observation_state=ol.OBSERVATION_FINAL,
            source_identity=f"player_gameweeks:{late_event}",
            source_payload_sha256=newer_live_record["payload_sha256"],
            fetch_run_id=newer_unbound_run["run_id"],
            archive_capture_id=newer_live_record["capture_id"],
            captured_at=newer_unbound_run["finished_at"],
        )
        conn.commit()
        selected_newer = ol.select_capture([
            row for row in ol.observation_captures(
                conn, grain=ol.GRAIN_PLAYER_EVENT, event=late_event, player_id=superseded_player,
            )
            if row.get("observation_state") == ol.OBSERVATION_FINAL
            and row.get("source_name") == "player_gameweeks_final"
            and row.get("source_identity") == f"player_gameweeks:{late_event}"
        ])
        assert selected_newer is not None
        assert int(selected_newer["fetch_run_id"]) == newer_unbound_run["run_id"]
        assert selected_newer["archive_capture_id"] == newer_live_record["capture_id"]

        def retained_maturation_artifact_bytes():
            paths = [
                *evidence_root.glob("chip-outcome-*.json"),
                *evidence_root.glob("chip-causal-evidence-*.json"),
                *evidence_ops._metadata_root(evidence_root).glob("maturation-*.json"),
            ]
            return {str(path.relative_to(evidence_root)): path.read_bytes() for path in paths}

        before_unbound_refusal = retained_maturation_artifact_bytes()
        with monkeypatch.context() as scoped:
            scoped.setattr(cal, "finalize_causal_observation", spy_finalizer)
            with pytest.raises(evidence_ops.ChipEvidenceError, match="not uniquely bound"):
                evidence_ops.mature_observation(
                    conn,
                    db_path=tmp_path / "origin.db", entry_id=241392,
                    generation_id=generation.generation_id, decision_id=decision_id,
                    route_id=route_id, planning_event=4, cutoff=generation.cutoff,
                    action="FH", observation_id="fh-source-forecast-1",
                    realization_event=schedule_events[0], evidence_root=evidence_root,
                )
        assert not finalizer_calls
        assert retained_maturation_artifact_bytes() == before_unbound_refusal

        # A later genuine complete archive and its normal public capture receipts
        # bind the latest rows, allowing the same retained observation to mature.
        recovery_run = _archive_future_official_state(
            conn, capture_root, schedule_events, final_events=schedule_events,
            live_players_by_event=players_by_event,
            observed_after=newer_unbound_run["finished_at"],
        )
        for event in schedule_events:
            evidence_ops.capture_outcome(
                conn, **{**capture_kwargs, "fetch_run_id": recovery_run["run_id"]},
                realization_event=event,
            )
        mature_kwargs = {
            "db_path": tmp_path / "origin.db", "entry_id": 241392,
            "generation_id": generation.generation_id, "decision_id": decision_id,
            "route_id": route_id, "planning_event": 4, "cutoff": generation.cutoff,
            "action": "FH", "observation_id": "fh-source-forecast-1",
            "realization_event": schedule_events[0], "evidence_root": evidence_root,
        }
        # Build the canonical complete pair exactly once, then test the state
        # where one immutable member survived but receipt publication did not.
        snapshot_conn = gs._open_generation_snapshot(generation)
        try:
            pinned_pool = cu.load_pool(snapshot_conn)
            source_positions = {
                int(player_id): str(row["position"])
                for player_id, row in pinned_pool.items()
                if row.get("position") is not None
            }
        finally:
            snapshot_conn.close()
        with monkeypatch.context() as scoped:
            scoped.setattr(cal, "finalize_causal_observation", spy_finalizer)
            finalized_pair = cal.finalize_causal_observation(
                conn,
                parent["causal"],
                store_conn=conn,
                realization_event=schedule_events[0],
                evidence_root=evidence_root,
                evidence_verifier=lambda reference: evidence_ops._flat_artifact(
                    evidence_root, reference,
                )[1],
                source_position_resolver=lambda _record: {
                    str(key): value for key, value in source_positions.items()
                },
            )
        assert len(finalizer_calls) == 1
        outcome_path = Path(finalized_pair["outcome_path"])
        causal_evidence_path = Path(finalized_pair["causal_evidence_path"])
        original_outcome_bytes = outcome_path.read_bytes()
        original_causal_bytes = causal_evidence_path.read_bytes()
        parked_outcome = tmp_path / outcome_path.name
        outcome_path.rename(parked_outcome)
        try:
            before_partial_refusal = retained_maturation_artifact_bytes()
            with monkeypatch.context() as scoped:
                scoped.setattr(cal, "finalize_causal_observation", spy_finalizer)
                with pytest.raises(evidence_ops.ChipEvidenceError, match="partial or ambiguous"):
                    evidence_ops.mature_observation(conn, **mature_kwargs)
            assert len(finalizer_calls) == 1
            assert retained_maturation_artifact_bytes() == before_partial_refusal
        finally:
            parked_outcome.rename(outcome_path)
        assert outcome_path.read_bytes() == original_outcome_bytes
        assert causal_evidence_path.read_bytes() == original_causal_bytes

        with monkeypatch.context() as scoped:
            scoped.setattr(cal, "finalize_causal_observation", spy_finalizer)
            matured = evidence_ops.mature_observation(conn, **mature_kwargs)
        assert len(finalizer_calls) == 1
        assert matured["receipt"]["forecast_available_at"] == forecast["receipt"]["forecast_available_at"]
        assert matured["receipt"]["forecast_made_at"] == parent["forecast"]["made_at"]
        verified_capture_artifacts = evidence_ops._verified_horizon_capture_manifest(
            conn, evidence_root=evidence_root, identity=identity, parent=parent,
            observation_id="fh-source-forecast-1", action="FH",
        )
        assert matured["receipt"]["capture_receipt_manifest"] == verified_capture_artifacts[1]
        assert matured["receipt"]["capture_manifest"] == verified_capture_artifacts[2]
        assert matured["calibration_row"]["forecast_available_at"] == forecast["receipt"]["forecast_available_at"]
        outcome = evidence_ops._read_json(evidence_root / matured["receipt"]["outcome_ref"])
        assert outcome["realization_events"] == schedule_events
        assert {
            int(row["event"]) for row in outcome["source"]["captures"]
        } == set(schedule_events)
        assert len(outcome["source"]["captures"]) == len(matured["receipt"]["capture_manifest"])

        # The common retained-proof loader is also the first gate for public
        # forecast retry/recovery, official capture, and maturation. Exercise
        # resealed adversarial copies in memory so the retained files stay
        # immutable while each public consumer refuses before side effects.
        def evidence_tree_bytes():
            return {
                str(path.relative_to(evidence_root)): path.read_bytes()
                for path in evidence_root.rglob("*") if path.is_file()
            }

        consumer_tampers = (
            (
                "legacy-cd0-missing-proof", remove_zero_hit_proof,
                "source-replayed zero-hit proof refused",
                "source-replayed zero-hit proof refused",
            ),
            (
                "malformed-witness", malformed_hit_cost_witness,
                "source-replayed zero-hit proof refused.*missing or unsupported",
                "source-replayed zero-hit proof refused.*missing or unsupported",
            ),
            (
                "resealed-selected-route", alter_selected_play_route,
                "selected optimizer family identity",
                "selected optimizer family identity",
            ),
            (
                "resealed-schedule", remove_late_valuation_event,
                "valuation-state event set differs from certified action horizons",
                "forecast retry differs from the immutable operation intent",
            ),
        )
        forecast_receipt_path = metadata_root / f"forecast-{operation_id}.json"
        for case_name, mutate, loader_match, recovery_match in consumer_tampers:
            overrides = _tampered_forecast_loader_artifacts(
                evidence_ops, evidence_root, operation_id, mutate,
            )
            if case_name == "legacy-cd0-missing-proof":
                legacy_publication = overrides[publication_path]
                legacy_receipt = overrides[metadata_root / f"forecast-{operation_id}.json"]
                assert "hit_cost_proof_sha256" not in legacy_publication
                assert "hit_cost_proof_sha256" not in legacy_receipt
                assert all(
                    "hit_cost_witness" not in report
                    for report in legacy_publication["assembly_reports"]
                )
            normalized_overrides = {
                Path(path).resolve(): value for path, value in overrides.items()
            }
            original_read_json = evidence_ops._read_json

            def read_tampered_json(path):
                resolved = Path(path).resolve()
                if resolved in normalized_overrides:
                    return normalized_overrides[resolved]
                return original_read_json(path)

            files_before = evidence_tree_bytes()
            db_changes_before = conn.total_changes
            captures_before = int(conn.execute(
                "SELECT COUNT(*) FROM outcome_observation_captures",
            ).fetchone()[0])
            finalizer_calls = []
            with monkeypatch.context() as scoped:
                scoped.setattr(evidence_ops, "_read_json", read_tampered_json)
                with pytest.raises(evidence_ops.ChipEvidenceError, match=loader_match):
                    horizon_ops.build_future_action_forecast(
                        conn, identity=identity, action="FH",
                        observation_id="fh-source-forecast-1", entry_id=241392,
                        continuation_generation_id=continuation.generation_id,
                        value_generation_ids={}, expiry_event=5,
                        coverage_product=coverage, evidence_root=evidence_root,
                        cache_dir=cache_dir,
                    )
                with pytest.raises(evidence_ops.ChipEvidenceError, match=loader_match):
                    evidence_ops.capture_outcome(
                        conn,
                        **{**capture_kwargs, "fetch_run_id": recovery_run["run_id"]},
                        realization_event=schedule_events[0],
                    )

                def forbid_tampered_finalization(*args, **kwargs):
                    finalizer_calls.append((args, kwargs))
                    raise AssertionError("invalid retained forecast reached canonical finalizer")

                scoped.setattr(cal, "finalize_causal_observation", forbid_tampered_finalization)
                with pytest.raises(evidence_ops.ChipEvidenceError, match=loader_match):
                    evidence_ops.mature_observation(conn, **mature_kwargs)
            assert not finalizer_calls
            assert conn.total_changes == db_changes_before
            assert int(conn.execute(
                "SELECT COUNT(*) FROM outcome_observation_captures",
            ).fetchone()[0]) == captures_before
            assert evidence_tree_bytes() == files_before

            # Exercise the distinct receipt-loss recovery branch. Its direct
            # publication helper and public issuer must reject before deriving
            # or writing the deterministic receipt; restore the real receipt
            # bytes after each isolated adversarial copy.
            retained_forecast_receipt = forecast_receipt_path.read_bytes()
            forecast_receipt_path.unlink()
            assert not forecast_receipt_path.exists()
            recovery_files_before = evidence_tree_bytes()
            recovery_db_changes_before = conn.total_changes
            recovery_capture_count_before = int(conn.execute(
                "SELECT COUNT(*) FROM outcome_observation_captures",
            ).fetchone()[0])
            reconstruction_calls = []
            receipt_writes = []
            recovery_finalizer_calls = []

            def forbid_receipt_reconstruction(*args, **kwargs):
                reconstruction_calls.append((args, kwargs))
                raise AssertionError("invalid publication reached deterministic reconstruction")

            def forbid_recovery_receipt_write(path, value):
                receipt_writes.append((path, value))
                raise AssertionError("invalid publication wrote the deterministic forecast receipt")

            def forbid_recovery_finalization(*args, **kwargs):
                recovery_finalizer_calls.append((args, kwargs))
                raise AssertionError("invalid publication reached outcome finalization")

            try:
                with monkeypatch.context() as scoped:
                    scoped.setattr(evidence_ops, "_read_json", read_tampered_json)
                    scoped.setattr(
                        evidence_ops, "_issued_forecast_artifacts",
                        forbid_receipt_reconstruction,
                    )
                    scoped.setattr(evidence_ops, "_write_new_json", forbid_recovery_receipt_write)
                    scoped.setattr(
                        cal, "finalize_causal_observation", forbid_recovery_finalization,
                    )
                    with pytest.raises(evidence_ops.ChipEvidenceError, match=loader_match):
                        horizon_ops._forecast_receipt_from_publication(
                            conn, identity=identity, origin_id=origin["origin"]["origin_id"],
                            operation_id=operation_id,
                            observation_id="fh-source-forecast-1", action="FH",
                            expiry_event=5,
                            publication=overrides[publication_path],
                            intent=overrides[metadata_root / f"forecast-intent-{operation_id}.json"],
                        )
                    with pytest.raises(evidence_ops.ChipEvidenceError, match=recovery_match):
                        horizon_ops.build_future_action_forecast(
                            conn, identity=identity, action="FH",
                            observation_id="fh-source-forecast-1", entry_id=241392,
                            continuation_generation_id=continuation.generation_id,
                            value_generation_ids={}, expiry_event=5,
                            coverage_product=coverage, evidence_root=evidence_root,
                            cache_dir=cache_dir,
                        )
                assert not reconstruction_calls
                assert not receipt_writes
                assert not recovery_finalizer_calls
                assert not forecast_receipt_path.exists()
                assert conn.total_changes == recovery_db_changes_before
                assert int(conn.execute(
                    "SELECT COUNT(*) FROM outcome_observation_captures",
                ).fetchone()[0]) == recovery_capture_count_before
                assert evidence_tree_bytes() == recovery_files_before
            finally:
                forecast_receipt_path.write_bytes(retained_forecast_receipt)
            assert forecast_receipt_path.read_bytes() == retained_forecast_receipt
            assert evidence_tree_bytes() == files_before

        # Removing only the deterministic receipt simulates publication failure:
        # the exact verified pair is recovered without calling the finalizer twice.
        maturation_path = evidence_ops._metadata_root(evidence_root) / (
            f"maturation-{evidence_ops._maturation_key('fh-source-forecast-1', schedule_events[0])}.json"
        )
        maturation_path.unlink()
        with monkeypatch.context() as scoped:
            scoped.setattr(
                cal, "finalize_causal_observation",
                lambda *_args, **_kwargs: pytest.fail("maturation recovery called the finalizer again"),
            )
            recovered = evidence_ops.mature_observation(conn, **mature_kwargs)
        assert recovered["receipt"]["outcome_ref"] == matured["receipt"]["outcome_ref"]
        assert recovered["receipt"]["causal_evidence_ref"] == matured["receipt"]["causal_evidence_ref"]

        # The deterministic receipt key intentionally omits action/origin.
        # Reusing the same observation/event under another action or explicit
        # origin must fail at origin binding and must not overwrite that key.
        retained_before_collisions = retained_maturation_artifact_bytes()
        for collision_kwargs in (
            {**mature_kwargs, "action": "WC"},
            {**mature_kwargs, "route_id": f"{route_id}-other-origin"},
        ):
            with monkeypatch.context() as scoped:
                scoped.setattr(
                    cal, "finalize_causal_observation",
                    lambda *_args, **_kwargs: pytest.fail("colliding identity reached finalization"),
                )
                with pytest.raises(evidence_ops.ChipEvidenceError):
                    evidence_ops.mature_observation(conn, **collision_kwargs)
        assert retained_maturation_artifact_bytes() == retained_before_collisions

        receipt_binding = matured["receipt"]["capture_receipt_manifest"][0]["receipts"][0]
        capture_receipt_path = evidence_ops._metadata_root(evidence_root) / receipt_binding["receipt_ref"]
        retained_capture_bytes = capture_receipt_path.read_bytes()
        retained_capture_receipt = evidence_ops._read_json(capture_receipt_path)
        tamper_cases = (
            ("action", "WC"),
            ("origin_id", "0" * 64),
            ("fetch_run_id", recovery_run["run_id"] + 999),
            ("available_at", "2099-01-01T00:00:00Z"),
        )
        for field, replacement in tamper_cases:
            body = dict(retained_capture_receipt)
            body.pop("schema", None)
            body.pop("receipt_sha256", None)
            body[field] = replacement
            capture_receipt_path.write_bytes(evidence_ops._json_bytes(
                evidence_ops._sealed(evidence_ops.CAPTURE_RECEIPT_SCHEMA, body),
            ))
            with pytest.raises(evidence_ops.ChipEvidenceError):
                evidence_ops.mature_observation(conn, **mature_kwargs)
            capture_receipt_path.write_bytes(retained_capture_bytes)

        # A newly appended, otherwise valid receipt is an extra event binding,
        # not permission to silently rewrite the already-matured manifest.
        extra_run = _archive_future_official_state(
            conn, capture_root, [schedule_events[0]],
            final_events=[schedule_events[0]],
            live_players_by_event={schedule_events[0]: players_by_event[schedule_events[0]]},
            observed_after=recovery_run["finished_at"],
        )
        evidence_ops.capture_outcome(
            conn,
            **{**capture_kwargs, "fetch_run_id": extra_run["run_id"]},
            realization_event=schedule_events[0],
        )
        with pytest.raises(
            evidence_ops.ChipEvidenceError,
            match="does not preserve forecast timing and the verified capture manifest",
        ):
            evidence_ops.mature_observation(conn, **mature_kwargs)
    finally:
        conn.close()


def test_public_future_wc_composer_refuses_missing_cache_without_materializing_worlds(
    tmp_path, monkeypatch, capsys,
):
    import generation_fixtures as gf
    from datetime import datetime, timedelta, timezone
    from fpl_brain import chip_decision as cd, chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_reservation_forecast as crf, chip_route_assembly as cra
    from fpl_brain import execution, generation_store as gs, monte_carlo

    # Give the two action-specific origins distinct, internally pinned windows:
    # FH is a GW5 opportunity, while the WC fixture retains its GW5-GW6 window.
    conn, generation, decision_id, route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=2, extended_events=(8, 9, 10, 11),
        chip_expiry_event=6, free_hit_expiry_event=5,
    )
    cache_dir = tmp_path / "must-stay-absent-wc-world-cache"
    try:
        route = cra.load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        runs = {event: generation.runs_for(event) for event in generation.events}
        for event in (8, 9, 10, 11):
            runs[event] = {
                str(row["model_family"]): int(row["id"])
                for row in conn.execute(
                    "SELECT id,model_family FROM projection_runs WHERE planning_event=?", (event,)
                )
            }
        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=4,
            planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="public_future_wc_cache_only_fixture",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            continuation = gs.certify_chip_reservation_generation(
                conn,
                chip_generation_id=generation.generation_id,
                planning_event=4,
                cutoff=generation.cutoff,
                events=tuple(range(4, 12)),
                runs_by_event=runs,
                snapshot=gf.fixture_snapshot(conn),
                controller=controller,
            )
        except BaseException as failure:
            controller.finish(execution.RUN_FAILED, str(failure))
            raise
        controller.finish(execution.RUN_COMPLETE)

        coverage = crf.build_reservation_coverage_product(
            conn,
            action=cd.CHIP_ACTION_WC,
            source_identity=assembly._source_identity(route, generation),
            expiry_event=6,
            product_generation_id=continuation.generation_id,
            wildcard_value_horizon_length=6,
        )
        value_controller = execution.ExecutionController(conn)
        value_controller.create_run(
            planning_event=5,
            planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="public_future_wc_value_fixture",
        )
        value_controller.start()
        value_controller.acquire_writer_lease()
        try:
            value_generation = gs.certify_future_wildcard_value_generation(
                conn,
                coverage_product=coverage,
                planning_event=5,
                controller=value_controller,
            )
        except BaseException as failure:
            value_controller.finish(execution.RUN_FAILED, str(failure))
            raise
        value_controller.finish(execution.RUN_COMPLETE)
        assert value_generation.events == (5, 6, 7, 8, 9, 10)
        value6_controller = execution.ExecutionController(conn)
        value6_controller.create_run(
            planning_event=6,
            planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="public_future_wc_value_gw6_fixture",
        )
        value6_controller.start()
        value6_controller.acquire_writer_lease()
        try:
            value_generation6 = gs.certify_future_wildcard_value_generation(
                conn,
                coverage_product=coverage,
                planning_event=6,
                controller=value6_controller,
            )
        except BaseException as failure:
            value6_controller.finish(execution.RUN_FAILED, str(failure))
            raise
        value6_controller.finish(execution.RUN_COMPLETE)
        assert value_generation6.events == (6, 7, 8, 9, 10, 11)
        assert coverage["required_product_events"] == list(range(4, 12))
        def forbid_world_simulation(*_args, **_kwargs):
            raise AssertionError("public WC composition attempted to materialize worlds")

        original_simulate = monte_carlo.simulate
        monkeypatch.setattr(monte_carlo, "simulate", forbid_world_simulation)

        with pytest.raises(FutureStateAssemblyError, match="certified continuation worlds for GW5 are unavailable"):
            assembly.build_future_wildcard_request(
                conn,
                decision_id=decision_id,
                generation_id=generation.generation_id,
                route_id=route_id,
                entry_id=241392,
                origin_cutoff=generation.cutoff,
                data_snapshot_sha256=str(generation.snapshot["sha256"]),
                target_event=5,
                season="2026/27",
                continuation_generation_id=continuation.generation_id,
                value_generation_id=value_generation.generation_id,
                cache_dir=cache_dir,
                expiry_event=6,
                coverage_product=coverage,
            )
        assert not cache_dir.exists()

        # Fixture cache creation happens outside the public call being verified.
        monkeypatch.setattr(monte_carlo, "simulate", original_simulate)
        _materialize_synthetic_fixture_caches(
            conn,
            continuation,
            decision_id,
            cache_dir,
            events=(5, 6, 7, 8, 9),
            windows=((5, 6, 7, 8), (6, 7, 8, 9)),
        )
        _materialize_synthetic_fixture_caches(
            conn,
            value_generation,
            decision_id,
            cache_dir,
            events=(5, 6, 7, 8, 9, 10),
            windows=((5, 6, 7, 8, 9, 10),),
        )
        _materialize_synthetic_fixture_caches(
            conn,
            value_generation6,
            decision_id,
            cache_dir,
            events=(6, 7, 8, 9, 10, 11),
            windows=((6, 7, 8, 9, 10, 11),),
        )
        monkeypatch.setattr(monte_carlo, "simulate", forbid_world_simulation)
        request, evidence = assembly.build_future_wildcard_request(
            conn,
            decision_id=decision_id,
            generation_id=generation.generation_id,
            route_id=route_id,
            entry_id=241392,
            origin_cutoff=generation.cutoff,
            data_snapshot_sha256=str(generation.snapshot["sha256"]),
            target_event=5,
            season="2026/27",
            continuation_generation_id=continuation.generation_id,
            value_generation_id=value_generation.generation_id,
            cache_dir=cache_dir,
            expiry_event=6,
            coverage_product=coverage,
        )
        assert request.horizon.events == (5, 6, 7, 8, 9, 10)
        assert tuple(sorted(request.worlds_by_event)) == (5, 6, 7, 8, 9, 10)
        assert tuple(sorted(request.players[1].events)) == (5, 6, 7, 8, 9, 10)
        assert tuple(row.event for row in request.save_route.events) == (5, 6, 7, 8)
        assert type(request.event_start_free_transfers) is int
        from dataclasses import replace
        from fpl_brain import chip_evidence_ops as evidence_ops
        from fpl_brain import chip_horizon_evidence_ops as horizon_ops
        from fpl_brain import chip_reservation_calibration as cal

        wildcard_state = next(
            dict(row) for row in request.chip_availability
            if str(row.get("name") or "").strip().lower() == "wildcard"
        )
        assert horizon_ops._verified_target_chip_available(
            request, action="WC", event=5, expiry_event=6,
            window_start_event=int(wildcard_state["window_start_event"]),
        ) is True
        wrong_window_rows = tuple(
            {**dict(row), "window_stop_event": 5}
            if str(row.get("name") or "").strip().lower() == "wildcard"
            else dict(row)
            for row in request.chip_availability
        )
        with pytest.raises(evidence_ops.ChipEvidenceError, match="does not prove one available"):
            horizon_ops._verified_target_chip_available(
                replace(request, chip_availability=wrong_window_rows),
                action="WC", event=5, expiry_event=6,
                window_start_event=int(wildcard_state["window_start_event"]),
            )
        with pytest.raises(evidence_ops.ChipEvidenceError, match="does not prove one available"):
            horizon_ops._verified_target_chip_available(
                request, action="WC", event=5, expiry_event=6,
                window_start_event=int(wildcard_state["window_start_event"]) - 1,
            )
        assert evidence["horizon_events"] == [5, 6, 7, 8, 9, 10]
        assert all(row["status"] == "HIT" for row in evidence["world_cache_report"])
        from fpl_brain import chip_wildcard as wc

        evaluation = wc.evaluate_wildcard(request)
        assert evaluation.action == cd.CHIP_ACTION_WC
        assert evaluation.data_snapshot_bound is True
        assert evaluation.execution_permitted is False
        assert evaluation.mean_uplift is not None
        assert evaluation.uncertainty["uplift_is_deterministic"] is True
        assert len(evaluation.candidate_metrics["wildcard_squad"]) == 15
        assert evaluation.candidate_metrics["horizon_points"] >= 0.0

        # The production producer validates the four-event chip certificate
        # independently from the six-event value horizon. Recreate the former
        # assembly identity (all continuation events) and prove the producer
        # rejects it before evaluating the request.
        from fpl_brain import wildcard_production as wcp

        chip_generation = gs.load_generation(conn, continuation.generation_id)
        chip_per_event = chip_generation.manifest["per_event"]
        old_continuation_identity = gs._certification_identity_for_bundles(
            cutoff=str(chip_generation.cutoff),
            bundle_identities={
                str(event): str(chip_per_event[str(event)]["bundle_identity"])
                for event in chip_generation.events
            },
            snapshot_sha256=str(chip_generation.snapshot["sha256"]),
        )
        assert old_continuation_identity != request.horizon_binding.certification_identity
        bad_binding = replace(
            request.horizon_binding,
            certification_identity=old_continuation_identity,
        )
        bad_request = replace(
            request,
            horizon_binding=bad_binding,
            certification_identity=old_continuation_identity,
        )
        save_first = next(row for row in request.save_route.events if row.event == 5)
        reservation_state = {
            "squad_ids": list(save_first.squad_ids),
            "bank_tenths": int(save_first.bank_tenths),
            "purchase_price_tenths": {
                str(int(pid)): int(price)
                for pid, price in sorted(save_first.purchase_price_tenths.items())
            },
            "free_transfers": int(save_first.free_transfers),
            "event_start_free_transfers": int(save_first.free_transfers),
            "wildcard_available": True,
        }
        source_identity = assembly._source_identity(route, generation)
        producer_args = {
            "source_identity": source_identity,
            "event": 5,
            "expiry_event": 6,
            "chip_generation_id": continuation.generation_id,
            "value_generation_id": value_generation.generation_id,
            "coverage_product": coverage,
            "reservation_state": reservation_state,
            "made_at": "2026-09-19T11:00:10Z",
        }
        with pytest.raises(
            wcp.WildcardProductionError,
            match="does not preserve its four-event decision",
        ):
            wcp.build_future_wildcard_event_opportunity(
                conn, request=bad_request, **producer_args,
            )

        opportunity = wcp.build_future_wildcard_event_opportunity(
            conn, request=request, **producer_args,
        )
        assert opportunity["evaluator_identity"]["decision_horizon_events"] == [5, 6, 7, 8]
        assert opportunity["evaluator_identity"]["value_horizon_events"] == [5, 6, 7, 8, 9, 10]
        assert opportunity["evaluator_identity"]["uncertainty"]["uplift_is_deterministic"] is True
        play_schedule = opportunity["outcome_arms"]["play"]["valuation_schedule"]["events"]
        save_schedule = opportunity["outcome_arms"]["save"]["valuation_schedule"]["events"]
        assert [row["event"] for row in play_schedule] == list(range(5, 11))
        assert [row["event"] for row in save_schedule] == list(range(5, 11))
        assert all(row["lineup"] for row in play_schedule + save_schedule)
        for schedule in (play_schedule, save_schedule):
            assert all(
                {"squad_ids", "bank_tenths", "purchase_price_tenths", "free_transfers"}
                <= set(row["transfer_state"])
                for row in schedule
            )
        play_initial = play_schedule[0]["transfer_state"]
        for row in play_schedule:
            state = row["transfer_state"]
            assert state["squad_ids"] == play_initial["squad_ids"]
            assert state["bank_tenths"] == play_initial["bank_tenths"]
            assert state["purchase_price_tenths"] == play_initial["purchase_price_tenths"]
        from fpl_brain import season_rules as sr

        expected_play_ft = int(play_initial["free_transfers"])
        for row in play_schedule[1:]:
            expected_play_ft = sr.free_transfers_after_gameweek(request.rules, expected_play_ft, 0)
            assert row["transfer_state"]["free_transfers"] == expected_play_ft
        assert save_schedule[4]["state_source"] == "CARRIED_CANONICAL_TERMINAL_STATE"
        assert save_schedule[5]["state_source"] == "CARRIED_CANONICAL_TERMINAL_STATE"
        terminal_state = request.save_route.terminal_squad_ids
        terminal_basis = request.save_route.terminal_purchase_price_tenths
        terminal_bank = request.save_route.terminal_bank_tenths
        terminal_ft = request.save_route.terminal_free_transfers
        expected_save_ft = int(terminal_ft)
        for row in save_schedule[4:]:
            expected_save_ft = sr.free_transfers_after_gameweek(request.rules, expected_save_ft, 0)
            state = row["transfer_state"]
            assert tuple(state["squad_ids"]) == tuple(terminal_state)
            assert state["bank_tenths"] == terminal_bank
            assert state["purchase_price_tenths"] == {
                str(pid): price for pid, price in sorted(terminal_basis.items())
            }
            assert state["free_transfers"] == expected_save_ft
            assert row["lineup"]

        _archive_future_official_state(conn, tmp_path / "origin-raw", [5, 6, 7, 8, 9, 10, 11])
        from fpl_brain import chip_horizon_evidence_ops as horizon_ops
        from fpl_brain import chip_evidence_ops as evidence_ops

        evidence_root = tmp_path / "wc-operator-evidence"
        identity = evidence_ops._required_identity(
            db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            evidence_root=evidence_root,
        )
        origin = horizon_ops.register_horizon_origin(
            conn,
            db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            action="WC", evidence_root=evidence_root,
        )
        assert origin["origin"]["chip_eligibility"]["expiry_event"] == 6
        fh_origin = horizon_ops.register_horizon_origin(
            conn,
            db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            action="FH", evidence_root=evidence_root,
        )
        assert fh_origin["origin"]["chip_eligibility"]["expiry_event"] == 5
        cli = _load_chip_evidence_cli()
        schema_version_before = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        app_schema_version_before = str(conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'",
        ).fetchone()[0])
        cli_common = [
            "--db", str(tmp_path / "origin.db"), "--entry-id", "241392",
            "--generation-id", generation.generation_id, "--decision-id", decision_id,
            "--route-id", route_id, "--planning-event", "4",
            "--origin-cutoff", generation.cutoff, "--evidence-root", str(evidence_root),
        ]
        assert cli.main([
            "register-horizon-origin", *cli_common, "--action", "WC",
        ]) == 0
        assert json.loads(capsys.readouterr().out)["receipt"]["action"] == "WC"
        operation_id = evidence_ops._forecast_operation_id(
            evidence_ops._sha256(evidence_ops._origin_key(identity)), "WC", "wc-source-forecast-1",
        )
        _archive_future_official_state(
            conn, tmp_path / "origin-raw", [5, 6, 7, 8, 9, 10, 11], final_events=[10],
        )
        with pytest.raises(evidence_ops.ChipEvidenceError, match="GW10 is FINAL"):
            horizon_ops.build_future_action_forecast(
                conn, identity=identity, action="WC", observation_id="wc-source-forecast-1",
                entry_id=241392, continuation_generation_id=continuation.generation_id,
                value_generation_ids={5: value_generation.generation_id, 6: value_generation6.generation_id},
                expiry_event=6,
                coverage_product=coverage, evidence_root=evidence_root, cache_dir=cache_dir,
            )
        metadata_root = evidence_ops._metadata_root(evidence_root)
        assert not (metadata_root / f"forecast-intent-{operation_id}.json").exists()
        assert not (metadata_root / f"forecast-publication-{operation_id}.json").exists()
        assert not (metadata_root / f"forecast-{operation_id}.json").exists()

        _archive_future_official_state(conn, tmp_path / "origin-raw", [5, 6, 7, 8, 9, 10, 11])
        # Refuse an incomplete later, non-selected opportunity before retaining any event record.
        blocked_root = tmp_path / "wc-later-opportunity-refusal"
        blocked_identity = evidence_ops._required_identity(
            db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            evidence_root=blocked_root,
        )
        horizon_ops.register_horizon_origin(
            conn, db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            action="WC", evidence_root=blocked_root,
        )
        original_wc_builder = assembly.build_future_wildcard_request
        original_retain = crf.retain_event_opportunity_record
        retained_before_late_refusal = []
        def omit_later_witness(conn, **kwargs):
            request, report = original_wc_builder(conn, **kwargs)
            if int(kwargs.get("target_event") or -1) == 6:
                report.pop("hit_cost_witness", None)
            return request, report
        def track_event_retention(*args, **kwargs):
            retained_before_late_refusal.append(True)
            return original_retain(*args, **kwargs)
        with monkeypatch.context() as scoped:
            scoped.setattr(assembly, "build_future_wildcard_request", omit_later_witness)
            scoped.setattr(crf, "retain_event_opportunity_record", track_event_retention)
            with pytest.raises(evidence_ops.ChipEvidenceError, match="GW6 FH/WC hit-cost witness is missing"):
                horizon_ops.build_future_action_forecast(
                    conn, identity=blocked_identity, action="WC",
                    observation_id="wc-later-opportunity-refusal", entry_id=241392,
                    continuation_generation_id=continuation.generation_id,
                    value_generation_ids={5: value_generation.generation_id, 6: value_generation6.generation_id},
                    expiry_event=6, coverage_product=coverage,
                    evidence_root=blocked_root, cache_dir=cache_dir,
                )
        assert retained_before_late_refusal == []
        # The genuine two-opportunity WC lifecycle below verifies both before its first retain.
        wc_gate_calls = []
        original_wc_gate = assembly.verify_future_action_hit_cost_witnesses
        original_roll_builder = assembly._build_canonical_roll_route
        wc_play_h2_states = []
        def record_canonical_roll_start(start_state, **kwargs):
            if int(start_state.event) == 6:
                wc_play_h2_states.append(start_state)
            return original_roll_builder(start_state, **kwargs)
        monkeypatch.setattr(assembly, "_build_canonical_roll_route", record_canonical_roll_start)
        def audited_wc_gate(conn, **kwargs):
            if str(kwargs.get("action") or "").upper() != "WC":
                return original_wc_gate(conn, **kwargs)
            events = tuple(kwargs["expected_events"])
            assert events == (5, 6)
            assert tuple(int(row["event"]) for row in kwargs["opportunity_records"]) == events
            assert tuple(int(row["event"]) for row in kwargs["assembly_reports"]) == events
            proof = original_wc_gate(conn, **kwargs)
            wc_gate_calls.append(events)
            return proof
        monkeypatch.setattr(assembly, "verify_future_action_hit_cost_witnesses", audited_wc_gate)
        original_retain_after_wc_gate = crf.retain_event_opportunity_record
        def wc_retain_only_after_source_gate(*args, **kwargs):
            assert wc_gate_calls and all(events == (5, 6) for events in wc_gate_calls)
            return original_retain_after_wc_gate(*args, **kwargs)
        monkeypatch.setattr(
            crf, "retain_event_opportunity_record", wc_retain_only_after_source_gate,
        )
        assert cli.main([
            "horizon-forecast", *cli_common,
            "--action", "WC", "--observation-id", "wc-source-forecast-1",
            "--expiry-event", "6", "--continuation-generation-id", continuation.generation_id,
            "--cache-dir", str(cache_dir), "--value-horizon-length", "6",
            "--value-generation", f"5={value_generation.generation_id}",
            "--value-generation", f"6={value_generation6.generation_id}",
        ]) == 0
        forecast = json.loads(capsys.readouterr().out)
        assert wc_gate_calls and all(events == (5, 6) for events in wc_gate_calls)
        assert wc_play_h2_states
        play_rows = forecast["forecast"]["opportunities"][0]["artifact_payload"]["outcome_arms"]["play"]["valuation_schedule"]["events"]
        assert play_rows[0]["wildcard_active"] is True
        assert all(row["wildcard_active"] is False for row in play_rows[1:])
        gw6_play_state = next(row["transfer_state"] for row in play_rows if int(row["event"]) == 6)
        h2_state = wc_play_h2_states[0]
        assert isinstance(h2_state, ts.RouteState)
        assert type(h2_state.transfers_made_this_event) is int
        assert h2_state.transfers_made_this_event == 0
        assert sorted(h2_state.by_id()) == gw6_play_state["squad_ids"]
        assert h2_state.bank_tenths == gw6_play_state["bank_tenths"]
        assert {str(pid): player.purchase_price_tenths for pid, player in sorted(h2_state.by_id().items())} == (
            gw6_play_state["purchase_price_tenths"]
        )
        assert h2_state.free_transfers == gw6_play_state["free_transfers"]
        assert h2_state.event_start_free_transfers == gw6_play_state["event_start_free_transfers"]
        target_state, _ = assembly._project_future_manager_state(route, target_event=5, rules=request.rules)
        assert h2_state.chip_state == target_state.chip_state
        assert forecast["forecast"]["coverage_complete"] is True
        recovery_observation_id = "wc-publication-recovery"
        recovery_origin_id = evidence_ops._sha256(evidence_ops._origin_key(identity))
        recovery_operation_id = evidence_ops._forecast_operation_id(
            recovery_origin_id, "WC", recovery_observation_id,
        )
        recovery_receipt_path = metadata_root / f"forecast-{recovery_operation_id}.json"
        recovery_publication_path = metadata_root / f"forecast-publication-{recovery_operation_id}.json"
        original_write_new_json = evidence_ops._write_new_json
        def fail_recovery_receipt_write(path, value):
            if Path(path) == recovery_receipt_path:
                raise OSError("injected receipt-write loss after complete publication")
            return original_write_new_json(path, value)
        with monkeypatch.context() as scoped:
            scoped.setattr(evidence_ops, "_write_new_json", fail_recovery_receipt_write)
            with pytest.raises(OSError, match="injected receipt-write loss"):
                horizon_ops.build_future_action_forecast(
                    conn, identity=identity, action="WC",
                    observation_id=recovery_observation_id, entry_id=241392,
                    continuation_generation_id=continuation.generation_id,
                    value_generation_ids={5: value_generation.generation_id, 6: value_generation6.generation_id},
                    expiry_event=6, coverage_product=coverage,
                    evidence_root=evidence_root, cache_dir=cache_dir,
                )
        assert recovery_publication_path.is_file()
        assert not recovery_receipt_path.exists()
        gates_before_receipt_recovery = len(wc_gate_calls)
        with monkeypatch.context() as scoped:
            scoped.setattr(assembly, "build_future_wildcard_request", lambda *_a, **_k: pytest.fail(
                "publication recovery must not recompose or rescore the WC",
            ))
            recovered_publication = horizon_ops.build_future_action_forecast(
                conn, identity=identity, action="WC",
                observation_id=recovery_observation_id, entry_id=241392,
                continuation_generation_id=continuation.generation_id,
                value_generation_ids={5: value_generation.generation_id, 6: value_generation6.generation_id},
                expiry_event=6, coverage_product=coverage,
                evidence_root=evidence_root, cache_dir=cache_dir,
            )
        assert recovered_publication["recovered_publication"] is True
        assert recovery_receipt_path.is_file()
        assert len(wc_gate_calls) > gates_before_receipt_recovery
        assert forecast["forecast"]["covered_events"] == [5, 6]
        assert forecast["receipt"]["selected_event"] == 5
        assert forecast["receipt"]["action"] == cd.CHIP_ACTION_WC
        assert forecast["receipt"]["materialized_worlds"] is False
        assert forecast["receipt"]["valuation_state_events"] == [5, 6, 7, 8, 9, 10, 11]
        gw6_assembly = next(row for row in forecast["assembly_reports"] if int(row["event"]) == 6)
        assert gw6_assembly["save_route_start_event_state"] == {
            "event": 6,
            "free_transfers": 3,
            "event_start_free_transfers": 3,
        }
        # The accepted SAVE DTO's `free_transfers` is the post-event route value;
        # it is not the WC request's event-start FT. Preserve and test both.
        assert gw6_assembly["save_route_target_event_dto_free_transfers"] == 4
        assert gw6_assembly["save_route_start_event_state"]["event_start_free_transfers"] != (
            gw6_assembly["save_route_target_event_dto_free_transfers"]
        )
        assert evidence_ops._parse_utc(
            forecast["receipt"]["forecast_available_at"], name="test forecast_available_at",
        ) >= evidence_ops._parse_utc(forecast["receipt"]["issued_at"], name="test issued_at")
        assert forecast["cache_report"] and all(row["status"] == "HIT" for row in forecast["cache_report"])
        publication_path = evidence_ops._metadata_root(evidence_root) / f"forecast-publication-{operation_id}.json"
        publication = evidence_ops._read_json(publication_path)
        assert [row["event"] for row in publication["valuation_prospective_availability"]] == [5, 6, 7, 8, 9, 10, 11]
        intent = evidence_ops._read_json(
            evidence_ops._metadata_root(evidence_root) / f"forecast-intent-{operation_id}.json",
        )
        publication_body = dict(publication)
        publication_body.pop("receipt_sha256", None)
        invalid_publication = evidence_ops._sealed(
            evidence_ops.FORECAST_PUBLICATION_SCHEMA,
            {**publication_body, "forecast_available_at": ""},
        )
        with pytest.raises(evidence_ops.ChipEvidenceError, match="forecast_available_at"):
            horizon_ops._forecast_receipt_from_publication(
                conn, identity=identity, origin_id=origin["origin"]["origin_id"],
                operation_id=operation_id, observation_id="wc-source-forecast-1", action="WC",
                expiry_event=6, publication=invalid_publication, intent=intent,
            )
        issued_arms = forecast["forecast"]["opportunities"][0]["artifact_payload"]["outcome_arms"]
        assert len(issued_arms["play"]["valuation_schedule"]["events"]) == 6
        assert len(issued_arms["save"]["valuation_schedule"]["events"]) == 6
        monkeypatch.setattr(wc, "evaluate_wildcard", lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("identical forecast retry rescored the WC"),
        ))
        assert cli.main([
            "horizon-forecast", *cli_common,
            "--action", "WC", "--observation-id", "wc-source-forecast-1",
            "--expiry-event", "6", "--continuation-generation-id", continuation.generation_id,
            "--cache-dir", str(cache_dir), "--value-horizon-length", "6",
            "--value-generation", f"5={value_generation.generation_id}",
            "--value-generation", f"6={value_generation6.generation_id}",
        ]) == 0
        replay = json.loads(capsys.readouterr().out)
        assert replay["reused"] is True
        assert replay["receipt"] == forecast["receipt"]
        assert replay["receipt"]["forecast_available_at"] == forecast["receipt"]["forecast_available_at"]

        # Retain a second, independently verified FH forecast for the same
        # observation ID and selected event. Both action-specific origins and
        # forecasts are genuine; their deterministic maturation key therefore
        # collides by design, exercising receipt binding rather than missing
        # origin/forecast refusal.
        fh_coverage = crf.build_reservation_coverage_product(
            conn,
            action=cd.CHIP_ACTION_FH,
            source_identity=assembly._source_identity(route, generation),
            expiry_event=5,
            product_generation_id=continuation.generation_id,
        )
        fh_forecast = horizon_ops.build_future_action_forecast(
            conn, identity=identity, action="FH", observation_id="wc-source-forecast-1",
            entry_id=241392, continuation_generation_id=continuation.generation_id,
            value_generation_ids={}, expiry_event=5, coverage_product=fh_coverage,
            evidence_root=evidence_root, cache_dir=cache_dir,
        )
        assert fh_forecast["receipt"]["selected_event"] == forecast["receipt"]["selected_event"] == 5

        parent = evidence_ops._load_forecast_origin(
            conn, identity=identity, action="WC", observation_id="wc-source-forecast-1",
        )[5]
        players_by_event = evidence_ops._schedule_player_map(parent["causal"], "WC")
        schedule_events = list(players_by_event)
        selected_event = int(forecast["receipt"]["selected_event"])
        assert schedule_events == list(range(selected_event, selected_event + 6))
        fh_parent = evidence_ops._load_forecast_origin(
            conn, identity=identity, action="FH", observation_id="wc-source-forecast-1",
        )[5]
        fh_players_by_event = evidence_ops._schedule_player_map(fh_parent["causal"], "FH")
        assert int(fh_parent["receipt"]["selected_event"]) == selected_event
        archive_players_by_event = {
            event: sorted(set(players_by_event.get(event, ())) | set(fh_players_by_event.get(event, ())))
            for event in sorted(set(players_by_event) | set(fh_players_by_event))
        }
        capture_root = tmp_path / "wc-outcome-raw"
        observed_after = max(
            evidence_ops._parse_utc(
                forecast["receipt"]["forecast_available_at"],
                name="WC forecast_available_at",
            ),
            evidence_ops._parse_utc(
                fh_forecast["receipt"]["forecast_available_at"],
                name="FH forecast_available_at",
            ),
        ).isoformat().replace("+00:00", "Z")
        complete_run = _archive_future_official_state(
            conn, capture_root, archive_players_by_event,
            final_events=archive_players_by_event,
            live_players_by_event=archive_players_by_event,
            observed_after=observed_after,
        )
        capture_kwargs = {
            "db_path": tmp_path / "origin.db", "entry_id": 241392,
            "generation_id": generation.generation_id, "decision_id": decision_id,
            "route_id": route_id, "planning_event": 4, "cutoff": generation.cutoff,
            "action": "WC", "observation_id": "wc-source-forecast-1",
            "fetch_run_id": complete_run["run_id"], "raw_root": capture_root,
            "evidence_root": evidence_root,
        }
        assert cli.main([
            "capture-outcome", *cli_common, "--action", "WC",
            "--observation-id", "wc-source-forecast-1", "--realization-event", str(schedule_events[0]),
            "--fetch-run-id", str(complete_run["run_id"]), "--raw-root", str(capture_root),
        ]) == 0
        cli_capture_receipt = json.loads(capsys.readouterr().out)
        assert cli_capture_receipt["action"] == "WC"
        for event in schedule_events[1:]:
            evidence_ops.capture_outcome(conn, **capture_kwargs, realization_event=event)
        fh_capture_kwargs = {**capture_kwargs, "action": "FH"}
        for event in fh_players_by_event:
            evidence_ops.capture_outcome(conn, **fh_capture_kwargs, realization_event=event)
        captures_before_retry = conn.execute(
            "SELECT COUNT(*) FROM outcome_observation_captures",
        ).fetchone()[0]
        first_capture_receipt = evidence_ops.capture_outcome(
            conn, **capture_kwargs, realization_event=schedule_events[0],
        )
        assert first_capture_receipt["action"] == "WC"
        assert conn.execute(
            "SELECT COUNT(*) FROM outcome_observation_captures",
        ).fetchone()[0] == captures_before_retry
        mature_args = [
            "mature", *cli_common, "--action", "WC",
            "--observation-id", "wc-source-forecast-1",
            "--realization-event", str(schedule_events[0]),
        ]
        # Demonstrate that both legitimate action-specific origins, forecasts
        # and their full capture manifests verify before the shared receipt key
        # is exercised.
        wc_verified_parent = evidence_ops._load_forecast_origin(
            conn, identity=identity, action="WC", observation_id="wc-source-forecast-1",
        )[5]
        fh_verified_parent = evidence_ops._load_forecast_origin(
            conn, identity=identity, action="FH", observation_id="wc-source-forecast-1",
        )[5]
        wc_verified_manifest = evidence_ops._verified_horizon_capture_manifest(
            conn, evidence_root=evidence_root, identity=identity, parent=wc_verified_parent,
            observation_id="wc-source-forecast-1", action="WC",
        )
        fh_verified_manifest = evidence_ops._verified_horizon_capture_manifest(
            conn, evidence_root=evidence_root, identity=identity, parent=fh_verified_parent,
            observation_id="wc-source-forecast-1", action="FH",
        )
        assert wc_verified_parent["receipt"]["action"] == "WC"
        assert fh_verified_parent["receipt"]["action"] == "FH"
        assert wc_verified_manifest[1] and fh_verified_manifest[1]
        assert int(wc_verified_parent["receipt"]["selected_event"]) == int(
            fh_verified_parent["receipt"]["selected_event"]
        )
        assert cli.main(mature_args) == 0
        matured = json.loads(capsys.readouterr().out)
        assert matured["receipt"]["forecast_available_at"] == forecast["receipt"]["forecast_available_at"]
        assert matured["receipt"]["forecast_made_at"] == parent["forecast"]["made_at"]
        assert [row["event"] for row in matured["receipt"]["capture_manifest"]] == [
            event for event in schedule_events for _pid in players_by_event[event]
        ]
        assert matured["calibration_row"]["forecast_available_at"] == forecast["receipt"]["forecast_available_at"]
        outcome = evidence_ops._read_json(evidence_root / matured["receipt"]["outcome_ref"])
        assert outcome["realization_events"] == schedule_events
        assert {
            int(row["event"]) for row in outcome["source"]["captures"]
        } == set(schedule_events)

        # Exercise the WC completed-publication proof through every shared
        # consumer, using the valid lifecycle above and in-memory tampering.
        forecast_receipt_path = metadata_root / f"forecast-{operation_id}.json"
        def wc_evidence_tree_bytes():
            return {
                str(path.relative_to(evidence_root)): path.read_bytes()
                for path in evidence_root.rglob("*") if path.is_file()
            }

        def remove_wc_hit_cost_proof(_intent, publication_body, receipt_body):
            publication_body.pop("hit_cost_proof_sha256", None)
            receipt_body.pop("hit_cost_proof_sha256", None)
            for report in publication_body.get("assembly_reports") or ():
                report.pop("hit_cost_witness", None)

        def malformed_wc_hit_cost_witness(_intent, publication_body, _receipt):
            publication_body["assembly_reports"][0]["hit_cost_witness"] = []

        def reseal_altered_wc_save_route(_intent, publication_body, _receipt):
            witness = publication_body["assembly_reports"][0]["hit_cost_witness"]
            witness["save_route_actions"][0]["event"] += 1
            witness_body = dict(witness)
            witness_body.pop("witness_sha256", None)
            witness["witness_sha256"] = crf.canonical_sha256(witness_body)

        def mismatch_wc_proof_digest(_intent, publication_body, _receipt):
            publication_body["hit_cost_proof_sha256"] = "0" * 64

        wc_consumer_tampers = (
            (remove_wc_hit_cost_proof,
             "source-replayed zero-hit proof refused.*missing or unsupported"),
            (malformed_wc_hit_cost_witness,
             "source-replayed zero-hit proof refused.*missing or unsupported"),
            (reseal_altered_wc_save_route,
             "source-replayed zero-hit proof refused: GW5 SAVE route witness differs from canonical source replay"),
            (mismatch_wc_proof_digest,
             "publication hit-cost proof digest is absent or does not verify"),
        )
        original_read_json = evidence_ops._read_json
        for mutate_wc_proof, refusal_pattern in wc_consumer_tampers:
            tampered_wc = _tampered_forecast_loader_artifacts(
                evidence_ops, evidence_root, operation_id, mutate_wc_proof,
            )
            normalized_wc = {Path(path).resolve(): value for path, value in tampered_wc.items()}

            def read_tampered_wc_json(path):
                resolved = Path(path).resolve()
                if resolved in normalized_wc:
                    return normalized_wc[resolved]
                return original_read_json(path)

            evidence_before = wc_evidence_tree_bytes()
            db_changes_before = conn.total_changes
            capture_count_before = int(conn.execute(
                "SELECT COUNT(*) FROM outcome_observation_captures",
            ).fetchone()[0])
            finalizer_calls = []

            def forbidden_wc_finalizer(*args, **kwargs):
                finalizer_calls.append((args, kwargs))
                raise AssertionError("tampered WC proof reached the canonical finalizer")

            with monkeypatch.context() as scoped:
                scoped.setattr(evidence_ops, "_read_json", read_tampered_wc_json)
                scoped.setattr(assembly, "build_future_wildcard_request", lambda *_a, **_k: pytest.fail(
                    "completed WC proof validation recomposed the request",
                ))
                scoped.setattr(wc, "evaluate_wildcard", lambda *_a, **_k: pytest.fail(
                    "completed WC proof validation rescored the evaluator",
                ))
                with pytest.raises(evidence_ops.ChipEvidenceError, match=refusal_pattern):
                    horizon_ops.build_future_action_forecast(
                        conn, identity=identity, action="WC",
                        observation_id="wc-source-forecast-1", entry_id=241392,
                        continuation_generation_id=continuation.generation_id,
                        value_generation_ids={
                            5: value_generation.generation_id,
                            6: value_generation6.generation_id,
                        },
                        expiry_event=6, coverage_product=coverage,
                        evidence_root=evidence_root, cache_dir=cache_dir,
                    )
                with pytest.raises(evidence_ops.ChipEvidenceError, match=refusal_pattern):
                    evidence_ops.capture_outcome(
                        conn, **capture_kwargs, realization_event=selected_event,
                    )
                scoped.setattr(cal, "finalize_causal_observation", forbidden_wc_finalizer)
                with pytest.raises(evidence_ops.ChipEvidenceError, match=refusal_pattern):
                    evidence_ops.mature_observation(
                        conn, db_path=tmp_path / "origin.db", entry_id=241392,
                        generation_id=generation.generation_id, decision_id=decision_id,
                        route_id=route_id, planning_event=4, cutoff=generation.cutoff,
                        action="WC", observation_id="wc-source-forecast-1",
                        realization_event=selected_event, evidence_root=evidence_root,
                    )
            assert not finalizer_calls
            assert conn.total_changes == db_changes_before
            assert int(conn.execute(
                "SELECT COUNT(*) FROM outcome_observation_captures",
            ).fetchone()[0]) == capture_count_before
            assert wc_evidence_tree_bytes() == evidence_before

            # Re-run the same refusal through deterministic publication
            # recovery with its receipt absent; no reconstruction or write is
            # permitted, and the exact original receipt is restored afterward.
            original_receipt_bytes = forecast_receipt_path.read_bytes()
            forecast_receipt_path.unlink()
            recovery_files_before = wc_evidence_tree_bytes()
            recovery_db_changes_before = conn.total_changes
            receipt_writes = []

            def forbid_wc_receipt_write(path, value):
                receipt_writes.append((path, value))
                raise AssertionError("tampered WC publication wrote a recovered receipt")

            try:
                with monkeypatch.context() as scoped:
                    scoped.setattr(evidence_ops, "_read_json", read_tampered_wc_json)
                    scoped.setattr(evidence_ops, "_write_new_json", forbid_wc_receipt_write)
                    scoped.setattr(evidence_ops, "_issued_forecast_artifacts", lambda *_a, **_k: pytest.fail(
                        "tampered WC publication reconstructed forecast artifacts",
                    ))
                    scoped.setattr(assembly, "build_future_wildcard_request", lambda *_a, **_k: pytest.fail(
                        "WC publication recovery recomposed the request",
                    ))
                    scoped.setattr(wc, "evaluate_wildcard", lambda *_a, **_k: pytest.fail(
                        "WC publication recovery rescored the evaluator",
                    ))
                    with pytest.raises(evidence_ops.ChipEvidenceError, match=refusal_pattern):
                        horizon_ops.build_future_action_forecast(
                            conn, identity=identity, action="WC",
                            observation_id="wc-source-forecast-1", entry_id=241392,
                            continuation_generation_id=continuation.generation_id,
                            value_generation_ids={
                                5: value_generation.generation_id,
                                6: value_generation6.generation_id,
                            },
                            expiry_event=6, coverage_product=coverage,
                            evidence_root=evidence_root, cache_dir=cache_dir,
                        )
                assert not receipt_writes
                assert not forecast_receipt_path.exists()
                assert conn.total_changes == recovery_db_changes_before
                assert wc_evidence_tree_bytes() == recovery_files_before
            finally:
                forecast_receipt_path.write_bytes(original_receipt_bytes)
            assert forecast_receipt_path.read_bytes() == original_receipt_bytes


        # The first valid WC maturation owns the deterministic
        # (observation_id, realization_event) receipt key. A second valid FH
        # origin/forecast at that same key must refuse on the retained receipt's
        # action binding, without replacing evidence or invoking the finalizer.
        fh_mature_kwargs = {
            "db_path": tmp_path / "origin.db", "entry_id": 241392,
            "generation_id": generation.generation_id, "decision_id": decision_id,
            "route_id": route_id, "planning_event": 4, "cutoff": generation.cutoff,
            "action": "FH", "observation_id": "wc-source-forecast-1",
            "realization_event": selected_event, "evidence_root": evidence_root,
        }
        assert evidence_ops._maturation_key(
            "wc-source-forecast-1", selected_event,
        ) == evidence_ops._maturation_key("wc-source-forecast-1", selected_event)

        def retained_shared_key_bytes():
            paths = [
                *evidence_root.glob("chip-outcome-*.json"),
                *evidence_root.glob("chip-causal-evidence-*.json"),
                *metadata_root.glob("maturation-*.json"),
            ]
            return {str(path.relative_to(evidence_root)): path.read_bytes() for path in paths}

        wc_maturation_path = metadata_root / (
            f"maturation-{evidence_ops._maturation_key('wc-source-forecast-1', selected_event)}.json"
        )
        wc_maturation_bytes = wc_maturation_path.read_bytes()
        before_valid_identity_collision = retained_shared_key_bytes()
        finalizer_collision_calls = []

        def forbidden_collision_finalizer(*args, **kwargs):
            finalizer_collision_calls.append((args, kwargs))
            raise AssertionError("valid competing FH maturation reached the finalizer")

        with monkeypatch.context() as scoped:
            scoped.setattr(cal, "finalize_causal_observation", forbidden_collision_finalizer)
            with pytest.raises(evidence_ops.ChipEvidenceError, match="maturation receipt"):
                evidence_ops.mature_observation(conn, **fh_mature_kwargs)
        assert not finalizer_collision_calls
        assert wc_maturation_path.read_bytes() == wc_maturation_bytes
        assert retained_shared_key_bytes() == before_valid_identity_collision

        assert cli.main(mature_args) == 0
        mature_retry = json.loads(capsys.readouterr().out)
        assert mature_retry["receipt"] == matured["receipt"]
        assert int(conn.execute("PRAGMA schema_version").fetchone()[0]) == schema_version_before
        assert str(conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'",
        ).fetchone()[0]) == app_schema_version_before
    finally:
        conn.close()


def test_public_wc_issuer_replays_later_nonselected_paid_hit_before_retention(
    tmp_path, monkeypatch,
):
    import generation_fixtures as gf
    from datetime import datetime, timedelta, timezone
    from fpl_brain import chip_decision as cd, chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_wildcard as wc
    from fpl_brain import chip_evidence_ops as evidence_ops
    from fpl_brain import chip_horizon_evidence_ops as horizon_ops
    from fpl_brain import chip_reservation_calibration as cal
    from fpl_brain import chip_reservation_forecast as crf, chip_route_assembly as cra
    from fpl_brain import execution, generation_store as gs

    paid_root = tmp_path / "later-paid-hit-origin"
    paid_root.mkdir()
    conn, generation, decision_id, route_id = _make_verified_origin(
        paid_root, monkeypatch, event_start_ft=0, free_transfers=0,
        later_paid_hit_event=6, extended_events=(8, 9, 10, 11), chip_expiry_event=6,
    )
    cache_dir = paid_root / "wc-world-cache"
    evidence_root = paid_root / "wc-evidence"
    try:
        route = cra.load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        runs = {event: generation.runs_for(event) for event in generation.events}
        for event in (8, 9, 10, 11):
            runs[event] = {
                str(row["model_family"]): int(row["id"])
                for row in conn.execute(
                    "SELECT id,model_family FROM projection_runs WHERE planning_event=?", (event,)
                )
            }
        continuation_controller = execution.ExecutionController(conn)
        continuation_controller.create_run(
            planning_event=4, planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="public_wc_later_paid_hit_continuation_fixture",
        )
        continuation_controller.start()
        continuation_controller.acquire_writer_lease()
        try:
            continuation = gs.certify_chip_reservation_generation(
                conn, chip_generation_id=generation.generation_id, planning_event=4,
                cutoff=generation.cutoff, events=tuple(range(4, 12)), runs_by_event=runs,
                snapshot=gf.fixture_snapshot(conn), controller=continuation_controller,
            )
        except BaseException as failure:
            continuation_controller.finish(execution.RUN_FAILED, str(failure))
            raise
        continuation_controller.finish(execution.RUN_COMPLETE)
        coverage = crf.build_reservation_coverage_product(
            conn, action=cd.CHIP_ACTION_WC, source_identity=assembly._source_identity(route, generation),
            expiry_event=6, product_generation_id=continuation.generation_id,
            wildcard_value_horizon_length=6,
        )
        value_generations = {}
        for event in (5, 6):
            value_controller = execution.ExecutionController(conn)
            value_controller.create_run(
                planning_event=event, planning_cutoff=generation.cutoff,
                hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
                label=f"public_wc_later_paid_hit_value_gw{event}_fixture",
            )
            value_controller.start()
            value_controller.acquire_writer_lease()
            try:
                value = gs.certify_future_wildcard_value_generation(
                    conn, coverage_product=coverage, planning_event=event,
                    controller=value_controller,
                )
            except BaseException as failure:
                value_controller.finish(execution.RUN_FAILED, str(failure))
                raise
            value_controller.finish(execution.RUN_COMPLETE)
            assert value.events == tuple(range(event, event + 6))
            value_generations[event] = value
        # Fail quickly on missing fixtures, malformed rosters, empty position
        # frontiers, or an illegal value-greedy seed before materializing any
        # optimizer world-cache entries.
        value_seed_preflight = {
            event: _assert_wc_value_greedy_seed_before_world_build(
                conn, decision_id=decision_id, generation_id=generation.generation_id,
                route_id=route_id, entry_id=241392, origin_cutoff=generation.cutoff,
                data_snapshot_sha256=generation.snapshot["sha256"],
                target_event=event, season="2026/27",
                value_generation=value_generations[event],
            )
            for event in (5, 6)
        }
        assert len(value_seed_preflight[5]) == len(value_seed_preflight[6]) == 15
        _materialize_synthetic_fixture_caches(
            conn, continuation, decision_id, cache_dir, events=(5, 6, 7, 8, 9),
            windows=((5, 6, 7, 8), (6, 7, 8, 9)),
        )
        value_worlds = {}
        for event, value in value_generations.items():
            window = tuple(range(event, event + 6))
            materialized = _materialize_synthetic_fixture_caches(
                conn, value, decision_id, cache_dir, events=window, windows=(window,),
            )
            value_worlds[event] = materialized[(window, int(event))]
        from fpl_brain import route_optimizer as ro
        gw5_matrix, gw5_cache = value_worlds[5]
        gw6_matrix, gw6_cache = value_worlds[6]
        assert gw5_cache["generation_id"] == value_generations[5].generation_id
        assert gw6_cache["generation_id"] == value_generations[6].generation_id
        assert gw5_matrix[ro.MANAGER_MATRIX_IDENTITY_KEY] == gw5_cache["key"]
        assert gw6_matrix[ro.MANAGER_MATRIX_IDENTITY_KEY] == gw6_cache["key"]
        assert 45 in gw5_matrix["player_ids"] and 45 in gw6_matrix["player_ids"]
        assert gw5_matrix["minutes"][45] == [90.0] * int(gw5_matrix["worlds"])
        assert gw6_matrix["minutes"][45] == [90.0] * int(gw6_matrix["worlds"])
        assert sum(gw5_matrix["core"][45]) > sum(gw6_matrix["core"][45])
        _archive_future_official_state(
            conn, paid_root / "origin-raw", (5, 6, 7, 8, 9, 10, 11),
        )
        identity = evidence_ops._required_identity(
            db_path=paid_root / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            evidence_root=evidence_root,
        )
        horizon_origin = horizon_ops.register_horizon_origin(
            conn, db_path=paid_root / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            action="WC", evidence_root=evidence_root,
        )
        assert horizon_origin["origin"]["chip_eligibility"]["expiry_event"] == 6
        source_identity = assembly._source_identity(route, generation)
        request_by_event = {}
        original_wc_builder = assembly.build_future_wildcard_request
        def capture_wc_request(conn, **kwargs):
            request, report = original_wc_builder(conn, **kwargs)
            target = int(kwargs["target_event"])
            scores, _screen = wc.screen_players(request)
            frontier = wc.position_frontiers(request, scores)
            seed = wc._greedy_squad(request, scores, frontier, key="value")
            assert tuple(seed or ()) == value_seed_preflight[target]
            legal, problems = wc.is_legal_squad(request, seed or ())
            assert legal, problems
            request_by_event[target] = request
            return request, report
        monkeypatch.setattr(assembly, "build_future_wildcard_request", capture_wc_request)

        replay_state = {"inside_gate": False, "targets": [], "gw6_hits": []}
        original_save_builder = assembly._build_future_save_route
        def audit_source_save_replay(route, **kwargs):
            projection = original_save_builder(route, **kwargs)
            if replay_state["inside_gate"]:
                target = int(kwargs["target_event"])
                replay_state["targets"].append(target)
                for action in projection.partial.actions:
                    if int(action["event"]) == 6:
                        transition = action["transition"]
                        replay_state["gw6_hits"].append({
                            "opportunity": target, "hit_points": transition.hit_points,
                            "transfers": [
                                {"out": int(move.out_player_id), "in": int(move.in_player_id)}
                                for move in action["batch"].actions
                            ],
                        })
            return projection
        monkeypatch.setattr(assembly, "_build_future_save_route", audit_source_save_replay)
        gate_calls = []
        original_gate = assembly.verify_future_action_hit_cost_witnesses
        def audit_full_public_gate(conn, **kwargs):
            events = tuple(kwargs["expected_events"])
            reports = {int(row["event"]): row for row in kwargs["assembly_reports"]}
            records = {int(row["event"]): row for row in kwargs["opportunity_records"]}
            assert events == (5, 6)
            assert tuple(sorted(reports)) == tuple(sorted(records)) == events
            record6 = records[6]
            report6 = reports[6]
            witness6 = report6["hit_cost_witness"]
            assert record6["action"] == witness6["action"] == "WC"
            assert record6["event"] == witness6["event"] == 6
            assert record6["planning_event"] == 4
            assert record6["source_identity"] == witness6["source_identity"] == source_identity
            assert record6["coverage_product_sha256"] == coverage["product_sha256"]
            assert witness6["continuation_generation_id"] == continuation.generation_id
            assert witness6["value_generation_id"] == value_generations[6].generation_id
            evaluator6 = record6["evaluator_identity"]
            assert evaluator6["decision_horizon_events"] == [6, 7, 8, 9]
            assert evaluator6["value_horizon_events"] == [6, 7, 8, 9, 10, 11]
            assert evaluator6["source_identity"] == source_identity
            save_arm = record6["outcome_arms"]["save"]
            save_schedule = save_arm["valuation_schedule"]["events"]
            assert [row["event"] for row in save_schedule] == list(range(6, 12))
            assert save_schedule[0]["lineup"] == save_arm["lineup"]
            assert request_by_event[6].planning_event == 6
            assert request_by_event[6].data_snapshot_sha256 == generation.snapshot["sha256"]
            assert tuple(request_by_event[6].horizon.events) == tuple(range(6, 12))
            assert request_by_event[6].save_route.events[0].event == 6
            assert dict(request_by_event[6].save_route.events[0].policy) == save_schedule[0]["lineup"]
            selected_score = max(float(row["expected_incremental_points"]) for row in records.values())
            selected_event = next(
                event for event in events
                if float(records[event]["expected_incremental_points"]) == selected_score
            )
            assert selected_event == 5 and selected_event != 6
            gate_calls.append(events)
            replay_state["inside_gate"] = True
            try:
                return original_gate(conn, **kwargs)
            finally:
                replay_state["inside_gate"] = False
        monkeypatch.setattr(assembly, "verify_future_action_hit_cost_witnesses", audit_full_public_gate)

        retained = []
        def reject_any_retention(*args, **kwargs):
            retained.append((args, kwargs))
            raise AssertionError("paid-hit refusal must precede every retained forecast artifact")
        monkeypatch.setattr(crf, "retain_event_opportunity_record", reject_any_retention)
        monkeypatch.setattr(crf, "retain_reservation_forecast", reject_any_retention)
        monkeypatch.setattr(cal, "retain_causal_origin_observation", reject_any_retention)
        observation_id = "wc-later-nonselected-paid-hit"
        with pytest.raises(
            evidence_ops.ChipEvidenceError,
            match="source-replayed zero-hit proof refused before retention.*GW6.*hit cost 4",
        ):
            horizon_ops.build_future_action_forecast(
                conn, identity=identity, action="WC", observation_id=observation_id,
                entry_id=241392, continuation_generation_id=continuation.generation_id,
                value_generation_ids={
                    event: value.generation_id for event, value in value_generations.items()
                },
                expiry_event=6, coverage_product=coverage,
                evidence_root=evidence_root, cache_dir=cache_dir,
            )
        assert gate_calls == [(5, 6)]
        assert tuple(sorted(request_by_event)) == (5, 6)
        assert replay_state["targets"] == [5, 6]
        assert replay_state["gw6_hits"] == [
            {"opportunity": 5, "hit_points": 4, "transfers": [{"out": 43, "in": 13}, {"out": 14, "in": 44}]},
            {"opportunity": 6, "hit_points": 4, "transfers": [{"out": 43, "in": 13}, {"out": 14, "in": 44}]},
        ]
        assert retained == []
        operation_id = evidence_ops._forecast_operation_id(
            evidence_ops._sha256(evidence_ops._origin_key(identity)), "WC", observation_id,
        )
        metadata_root = evidence_ops._metadata_root(evidence_root)
        assert (metadata_root / f"forecast-intent-{operation_id}.json").is_file()
        assert not (metadata_root / f"forecast-publication-{operation_id}.json").exists()
        assert not (metadata_root / f"forecast-{operation_id}.json").exists()
    finally:
        conn.close()

def test_rapid_changed_archives_get_monotonic_observations_from_normalized_manifest(
    tmp_path, monkeypatch,
):
    from fpl_brain import raw_archive

    conn, _generation, _decision_id, _route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=2,
    )
    raw_root = tmp_path / "origin-raw"
    fixed_wall_clock_base = "2030-01-01T12:00:00Z"
    try:
        first = _archive_future_official_state(
            conn, raw_root, (4,), wall_clock_base_override=fixed_wall_clock_base,
        )
        second = _archive_future_official_state(
            conn, raw_root, (4,), final_events=(4,),
            wall_clock_base_override=fixed_wall_clock_base,
        )

        assert first["observed_at"] < second["observed_at"]
        run_ids = {str(first["run_id"]), str(second["run_id"])}
        manifest = raw_archive.load_manifest(raw_root)
        for source in ("bootstrap_static", "fixtures"):
            rows = sorted(
                (row for row in manifest
                 if row.get("source") == source and str(row.get("run_id")) in run_ids),
                key=lambda row: row["observed_at"],
            )
            assert len(rows) == 2
            assert rows[0]["observed_at"] == first["observed_at"]
            assert rows[1]["observed_at"] == second["observed_at"]
            assert rows[0]["payload_sha256"] != rows[1]["payload_sha256"]
    finally:
        conn.close()


def _optimizer_capture_union_from_actual_inputs(ro, call):
    """Mirror optimize's exact initial capture-union derivation on copied inputs."""
    import copy

    initial_state = copy.deepcopy(call["initial_state"])
    universe = copy.deepcopy(call["universe"])
    config = copy.deepcopy(call["config"])
    override = call.get("search_n_override")
    if override is not None:
        config = ro._with_n(config, int(override))

    owned_ids = [int(player.player_id) for player in initial_state.players]
    search_pool = ro.build_search_pool(universe, owned_ids, config)
    required_routes = tuple(call.get("required_routes") or ())
    inherited_routes = tuple((call.get("nested_prior") or {}).get("promoted") or ())
    forced_ids = set()
    required_player_ids = []
    inherited_player_ids = []
    for partial in required_routes:
        route_ids = tuple(sorted(int(pid) for pid in ro.route_player_ids(partial)))
        required_player_ids.append(route_ids)
        forced_ids.update(route_ids)
    for partial in inherited_routes:
        route_ids = tuple(sorted(int(pid) for pid in ro.route_player_ids(partial)))
        inherited_player_ids.append(route_ids)
        forced_ids.update(route_ids)

    union_ids = tuple(ro.union_player_ids(
        initial_state, sorted(set(search_pool["pool_ids"]) | forced_ids),
    ))
    assert union_ids == tuple(sorted(set(union_ids)))
    assert set(owned_ids).issubset(set(union_ids))
    return {
        "config": config,
        "initial_state_ids": tuple(sorted(owned_ids)),
        "tail_universe_ids": tuple(sorted({
            int(row["player_id"]) for row in universe["universe"]
        })),
        "search_pool_ids": tuple(int(pid) for pid in search_pool["pool_ids"]),
        "required_route_player_ids": tuple(required_player_ids),
        "inherited_route_player_ids": tuple(inherited_player_ids),
        "union_ids": union_ids,
    }


def _prepare_public_fh_optimizer_capture_caches(
    conn, *, generation, decision_id, route_id, continuation, coverage, cache_dir,
    prewarmed_caches, monkeypatch,
):
    """Probe exact PLAY optimizer inputs, then build only their certified caches."""
    from pathlib import Path
    from fpl_brain import chip_horizon_evidence_assembly as assembly
    from fpl_brain import route_optimizer as ro

    events = (6, 7, 8)
    full_pool_matrix = prewarmed_caches[((5, 6, 7, 8), 6)][0]
    full_pool_ids = tuple(sorted(int(pid) for pid in full_pool_matrix["player_ids"]))
    captured = {}
    probe_cache_calls = []
    original_optimize = ro.optimize
    original_builder = ro.build_event_worlds
    db_changes_before = conn.total_changes

    def observe_optimizer(**kwargs):
        assert not captured, "the canonical PLAY setup probe invoked optimize more than once"
        assert kwargs.get("allow_world_materialization") is False
        assert type(kwargs.get("allow_world_materialization")) is bool
        derived = _optimizer_capture_union_from_actual_inputs(ro, kwargs)
        config = derived["config"]
        generation_arg = kwargs["generation"]
        assert str(generation_arg.generation_id) == str(continuation.generation_id)
        assert tuple(int(event) for event in config.events) == events
        assert int(config.search_draws) == 2
        assert int(config.seed) == 62026
        assert kwargs.get("search_n_override") is None
        assert derived["union_ids"] != full_pool_ids, (
            "fixture must distinguish the optimizer's canonical capture union from its "
            "prewarmed full eligible-pool context cache"
        )
        captured.update({
            **derived,
            "generation_id": str(generation_arg.generation_id),
            "runs_by_event": {
                event: dict(generation_arg.runs_for(event)) for event in events
            },
            "allow_world_materialization": kwargs["allow_world_materialization"],
        })
        # Pass every original input through unchanged. This cache-only probe is
        # expected to refuse at its first absent exact optimizer cache.
        return original_optimize(**kwargs)

    def observe_builder(conn_arg, generation_arg, event_arg, union_arg, config_arg, **kwargs):
        if tuple(int(event) for event in config_arg.events) == events:
            event = int(event_arg)
            key = ro.world_cache_key(
                event=event,
                generation_id=generation_arg.generation_id,
                runs=generation_arg.runs_for(event),
                config=config_arg,
                union_ids=union_arg,
            )
            probe_cache_calls.append({
                "event": event,
                "generation_id": str(generation_arg.generation_id),
                "union_ids": tuple(sorted(int(pid) for pid in union_arg)),
                "config_events": tuple(int(value) for value in config_arg.events),
                "allow_materialization": kwargs.get("allow_materialization"),
                "cache_dir": str(Path(kwargs["cache_dir"]).resolve()),
                "key": key,
            })
        return original_builder(conn_arg, generation_arg, event_arg, union_arg, config_arg, **kwargs)

    monkeypatch.setattr(ro, "optimize", observe_optimizer)
    monkeypatch.setattr(ro, "build_event_worlds", observe_builder)
    probe_failure = None
    try:
        assembly.build_future_free_hit_request(
            conn,
            decision_id=decision_id,
            generation_id=generation.generation_id,
            route_id=route_id,
            entry_id=241392,
            origin_cutoff=generation.cutoff,
            data_snapshot_sha256=str(generation.snapshot["sha256"]),
            target_event=5,
            season="2026/27",
            continuation_generation_id=continuation.generation_id,
            cache_dir=cache_dir,
            expiry_event=5,
            coverage_product=coverage,
        )
    except assembly.FutureStateAssemblyError as failure:
        probe_failure = str(failure)
    finally:
        # The following certified cache construction is outside every
        # composer/optimizer invocation and outside the observation wrappers.
        monkeypatch.setattr(ro, "optimize", original_optimize)
        monkeypatch.setattr(ro, "build_event_worlds", original_builder)

    assert probe_failure is not None, "expected the cache-only setup probe to refuse"
    assert len(captured) > 0
    assert len(probe_cache_calls) == 1
    assert probe_cache_calls[0]["event"] == events[0]
    assert probe_cache_calls[0]["allow_materialization"] is False
    assert probe_cache_calls[0]["generation_id"] == str(continuation.generation_id)
    assert probe_cache_calls[0]["union_ids"] == captured["union_ids"]
    assert probe_cache_calls[0]["config_events"] == events
    assert probe_cache_calls[0]["cache_dir"] == str(Path(cache_dir).resolve())
    assert conn.total_changes == db_changes_before

    config = captured["config"]
    union_ids = captured["union_ids"]
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_keys = {}
    for event in events:
        assert dict(continuation.runs_for(event)) == captured["runs_by_event"][event]
        cache_keys[event] = ro.world_cache_key(
            event=event,
            generation_id=continuation.generation_id,
            runs=continuation.runs_for(event),
            config=config,
            union_ids=union_ids,
        )
    assert "CERTIFIED_WORLD_CACHE_REQUIRED" in probe_failure
    assert "MISS_ABSENT" in probe_failure
    assert cache_keys[events[0]] in probe_failure
    assert not (cache_root / f"{cache_keys[events[0]]}.json").exists()

    # Build only the exact optimizer capture union, directly from the certified
    # generation. No composer/optimizer call or altered production flag is used.
    for event in events:
        cache_path = cache_root / f"{cache_keys[event]}.json"
        assert not cache_path.exists()
        matrix, build_info = ro.build_event_worlds(
            conn, continuation, event, union_ids, config,
            cache_dir=cache_root, allow_materialization=True,
        )
        assert str(build_info["key"]) == cache_keys[event]
        assert str(build_info["generation_id"]) == str(continuation.generation_id)
        assert str(matrix.get(ro.MANAGER_MATRIX_IDENTITY_KEY) or "") == cache_keys[event]
        assert tuple(sorted(int(pid) for pid in matrix["player_ids"])) == union_ids
        loaded, read_info = ro.read_world_cache_entry(cache_path)
        assert loaded is not None
        assert read_info["cache"]["status"] == ro.CACHE_HIT
        assert tuple(sorted(int(pid) for pid in loaded["player_ids"])) == union_ids

    later_cache_calls = []
    later_matrix_observations = []

    def verify_later_cache_identity(conn_arg, generation_arg, event_arg, union_arg, config_arg, **kwargs):
        if tuple(int(event) for event in config_arg.events) == events:
            event = int(event_arg)
            assert event in events
            assert type(kwargs.get("allow_materialization")) is bool
            assert kwargs["allow_materialization"] is False
            assert str(generation_arg.generation_id) == str(continuation.generation_id)
            actual_union = tuple(sorted(int(pid) for pid in union_arg))
            assert actual_union == union_ids
            assert dict(generation_arg.runs_for(event)) == captured["runs_by_event"][event]
            assert Path(kwargs["cache_dir"]).expanduser().resolve() == cache_root
            actual_key = ro.world_cache_key(
                event=event,
                generation_id=generation_arg.generation_id,
                runs=generation_arg.runs_for(event),
                config=config_arg,
                union_ids=actual_union,
            )
            assert actual_key == cache_keys[event]
            later_cache_calls.append({
                "event": event, "union_ids": actual_union, "key": actual_key,
            })
        loaded = original_builder(conn_arg, generation_arg, event_arg, union_arg, config_arg, **kwargs)
        if tuple(int(value) for value in config_arg.events) == events:
            later_matrix_observations.append({
                "event": int(event_arg),
                "generation_id": str(generation_arg.generation_id),
                "union_ids": tuple(sorted(int(pid) for pid in union_arg)),
                "config_events": tuple(int(value) for value in config_arg.events),
                "allow_materialization": kwargs.get("allow_materialization"),
                "cache_dir": str(Path(kwargs["cache_dir"]).expanduser().resolve()),
                "key": later_cache_calls[-1]["key"],
                "matrix": loaded[0],
                "evidence": loaded[1],
            })
        return loaded

    monkeypatch.setattr(ro, "build_event_worlds", verify_later_cache_identity)
    return {
        "play_optimizer_union_ids": union_ids,
        "play_optimizer_config_events": tuple(int(event) for event in config.events),
        "play_optimizer_cache_keys": cache_keys,
        "play_optimizer_runs_by_event": captured["runs_by_event"],
        "play_optimizer_setup_probe": {
            "captured_inputs": captured,
            "wrapped_failure": probe_failure,
            "cache_calls": probe_cache_calls,
            "cache_materialization_basis": "the existing helper restores both wrappers before its direct certified prewarm loop",
        },
        "play_cache_loader_calls": later_cache_calls,
        "play_cache_matrix_observations": later_matrix_observations,
        "play_optimizer_tail_universe_ids": captured["tail_universe_ids"],
        "play_optimizer_search_pool_ids": captured["search_pool_ids"],
        "play_optimizer_initial_state_ids": captured["initial_state_ids"],
        "play_optimizer_required_route_player_ids": captured["required_route_player_ids"],
        "play_optimizer_inherited_route_player_ids": captured["inherited_route_player_ids"],
    }

def _prepare_public_fh_hit_gate_case(
    tmp_path, monkeypatch, *, later_paid_hit_event=None, paid_fh_play_event=None,
    historical_paid_hit_event=None, suppress_fh_play_standouts=False,
):
    import generation_fixtures as gf
    from datetime import datetime, timedelta, timezone
    from fpl_brain import chip_decision as cd, chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_evidence_ops as evidence_ops
    from fpl_brain import chip_horizon_evidence_ops as horizon_ops
    from fpl_brain import chip_reservation_forecast as crf, chip_route_assembly as cra
    from fpl_brain import execution, generation_store as gs

    conn, generation, decision_id, route_id = _make_verified_origin(
        tmp_path, monkeypatch, event_start_ft=0, free_transfers=0,
        extended_events=(8, 9, 10, 11), chip_expiry_event=5, free_hit_expiry_event=5,
        later_paid_hit_event=later_paid_hit_event, paid_fh_play_event=paid_fh_play_event,
        historical_paid_hit_event=historical_paid_hit_event,
        suppress_fh_play_standouts=suppress_fh_play_standouts,
    )
    try:
        route = cra.load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        runs = {event: generation.runs_for(event) for event in generation.events}
        for event in (8, 9, 10, 11):
            runs[event] = {
                str(row["model_family"]): int(row["id"])
                for row in conn.execute(
                    "SELECT id,model_family FROM projection_runs WHERE planning_event=?",
                    (event,),
                )
            }
        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=4, planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="fh_hit_cost_boundary_fixture",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            continuation = gs.certify_chip_reservation_generation(
                conn, chip_generation_id=generation.generation_id,
                planning_event=4, cutoff=generation.cutoff,
                events=tuple(range(4, 12)), runs_by_event=runs,
                snapshot=gf.fixture_snapshot(conn), controller=controller,
            )
        except BaseException as failure:
            controller.finish(execution.RUN_FAILED, str(failure))
            raise
        controller.finish(execution.RUN_COMPLETE)
        coverage = crf.build_reservation_coverage_product(
            conn, action=cd.CHIP_ACTION_FH,
            source_identity=assembly._source_identity(route, generation),
            expiry_event=5, product_generation_id=continuation.generation_id,
        )
        cache_dir = tmp_path / "fh-world-cache"
        prewarmed_caches = _materialize_synthetic_fixture_caches(
            conn, continuation, decision_id, cache_dir,
            events=(5, 6, 7, 8), windows=((5, 6, 7, 8),),
        )
        optimizer_cache = _prepare_public_fh_optimizer_capture_caches(
            conn,
            generation=generation,
            decision_id=decision_id,
            route_id=route_id,
            continuation=continuation,
            coverage=coverage,
            cache_dir=cache_dir,
            prewarmed_caches=prewarmed_caches,
            monkeypatch=monkeypatch,
        )
        raw_root = tmp_path / "origin-raw"
        _archive_future_official_state(conn, raw_root, (5, 6, 7, 8))
        evidence_root = tmp_path / "fh-evidence"
        identity = evidence_ops._required_identity(
            db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            evidence_root=evidence_root,
        )
        horizon_ops.register_horizon_origin(
            conn, db_path=tmp_path / "origin.db", entry_id=241392,
            generation_id=generation.generation_id, decision_id=decision_id,
            route_id=route_id, planning_event=4, cutoff=generation.cutoff,
            action="FH", evidence_root=evidence_root,
        )
        return {
            "conn": conn, "generation": generation, "decision_id": decision_id,
            "route_id": route_id, "continuation": continuation,
            "coverage": coverage, "cache_dir": cache_dir,
            "raw_root": raw_root, "evidence_root": evidence_root,
            "identity": identity,
            **optimizer_cache,
        }
    except BaseException:
        conn.close()
        raise


@pytest.mark.parametrize(
    ("later_paid_hit_event", "paid_fh_play_event", "expected_costs"),
    [
        (6, 6, {"PLAY": 4, "SAVE": 4}),
        (None, 6, {"PLAY": 4, "SAVE": 0}),
        (6, None, {"PLAY": 0, "SAVE": 4}),
    ],
)
def test_public_fh_full_source_replay_refuses_paid_play_and_save_schedules(
    tmp_path, monkeypatch, later_paid_hit_event, paid_fh_play_event, expected_costs,
):
    from fpl_brain import chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_evidence_ops as evidence_ops
    from fpl_brain import chip_horizon_evidence_ops as horizon_ops
    from fpl_brain import chip_reservation_calibration as cal
    from fpl_brain import (
        chip_reservation_forecast as crf, chip_route_assembly as route_assembly,
        monte_carlo, route_optimizer as ro,
    )

    context = _prepare_public_fh_hit_gate_case(
        tmp_path, monkeypatch, later_paid_hit_event=later_paid_hit_event,
        paid_fh_play_event=paid_fh_play_event,
        suppress_fh_play_standouts=True,
    )
    conn = context["conn"]
    try:
        optimizer_observations = []
        original_optimize = ro.optimize

        def capture_canonical_optimizer(**kwargs):
            observation = {"kwargs": kwargs}
            optimizer_observations.append(observation)
            try:
                result = original_optimize(**kwargs)
            except BaseException as exc:
                observation.update({
                    "outcome": "raised",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                })
                raise
            observation.update({"outcome": "returned", "result": result})
            return result

        # The cache-miss setup probe has already run and restored its wrapper in
        # _prepare_public_fh_optimizer_capture_caches. Observe only the single
        # canonical composer call below; pass exact inputs through once, return
        # the identical result on success, and preserve the original exception.
        monkeypatch.setattr(ro, "optimize", capture_canonical_optimizer)
        built = {}
        original_builder = assembly.build_future_free_hit_request

        def capture_builder(conn, **kwargs):
            request, report = original_builder(conn, **kwargs)
            built[int(kwargs["target_event"])] = (request, report)
            return request, report

        monkeypatch.setattr(assembly, "build_future_free_hit_request", capture_builder)
        gate_inputs = []
        hit_charge_observations = []
        gate_delegations = []
        original_gate = assembly.verify_future_action_hit_cost_witnesses

        def capture_full_gate(conn, **kwargs):
            gate_inputs.append(kwargs)
            assert tuple(kwargs["expected_events"]) == (5,)
            witness = kwargs["assembly_reports"][0]["hit_cost_witness"]
            for arm, actions in (
                ("PLAY", witness["play_route_record"]["actions"]),
                ("SAVE", witness["save_route_actions"]),
            ):
                rows = [row for row in actions if int(row.get("event", -1)) == 6]
                raw = rows[0].get("hit_points") if len(rows) == 1 else None
                hit_charge_observations.append({
                    "arm": arm,
                    "event": 6,
                    "action_count": len(rows),
                    "observed_hit_points": raw if type(raw) is int else None,
                    "observed_type": type(raw).__name__ if rows else "missing",
                    "source_route_id": (
                        witness["play_route_record"].get("route_id")
                        if arm == "PLAY" else "SAVE_ROUTE_WITNESS"
                    ),
                })
            for arm, expected in expected_costs.items():
                actions = (
                    witness["play_route_record"]["actions"]
                    if arm == "PLAY" else witness["save_route_actions"]
                )
                arm_rows = [
                    row for row in actions
                    if int(row.get("event", -1)) == 6
                ]
                assert len(arm_rows) == 1, (
                    f"Generated {arm} arm must contain exactly one GW6 action"
                )
                arm_cost = arm_rows[0]["hit_points"]
                assert type(arm_cost) is int, (
                    f"Generated {arm} arm GW6 charge must be an integer"
                )
                assert arm_cost == expected, (
                    f"Generated {arm} arm GW6 hit cost {arm_cost} "
                    f"does not match expected {expected}"
                )
            delegation = {
                "action": kwargs["action"],
                "expected_events": list(kwargs["expected_events"]),
                "continuation_generation_id": kwargs["continuation_generation_id"],
                "witness_sha256": witness.get("witness_sha256"),
            }
            try:
                result = original_gate(conn, **kwargs)
            except BaseException as exc:
                gate_delegations.append({
                    **delegation, "outcome": "raised",
                    "exception_type": type(exc).__name__, "message": str(exc),
                })
                raise
            gate_delegations.append({**delegation, "outcome": "returned", "result": result})
            return result

        monkeypatch.setattr(assembly, "verify_future_action_hit_cost_witnesses", capture_full_gate)
        monkeypatch.setattr(
            monte_carlo, "simulate",
            lambda *_args, **_kwargs: pytest.fail("cache-only FH composition attempted world construction"),
        )
        retained = []

        def forbid_retention(*args, **kwargs):
            retained.append((args, kwargs))
            raise AssertionError("paid-hit source proof must fail before retention")

        monkeypatch.setattr(crf, "retain_event_opportunity_record", forbid_retention)
        monkeypatch.setattr(crf, "retain_reservation_forecast", forbid_retention)
        monkeypatch.setattr(cal, "retain_causal_origin_observation", forbid_retention)
        observation_id = "fh-full-source-hit-cost"
        db_changes_before = conn.total_changes
        with pytest.raises(evidence_ops.ChipEvidenceError) as failure:
            horizon_ops.build_future_action_forecast(
                conn, identity=context["identity"], action="FH",
                observation_id=observation_id, entry_id=241392,
                continuation_generation_id=context["continuation"].generation_id,
                value_generation_ids={}, expiry_event=5, coverage_product=context["coverage"],
                evidence_root=context["evidence_root"], cache_dir=context["cache_dir"],
            )

        assert len(gate_delegations) == 1
        assert gate_delegations[0]["outcome"] == "raised"
        assert gate_delegations[0]["exception_type"] == "FutureStateAssemblyError"
        message = str(failure.value)
        assert "source-replayed zero-hit proof refused before retention" in message
        assert len(gate_inputs) == 1
        assert [int(row["event"]) for row in gate_inputs[0]["assembly_reports"]] == [5]
        assert [int(row["event"]) for row in gate_inputs[0]["opportunity_records"]] == [5]
        assert set(built) == {5}
        _request, report = built[5]
        witness = report["hit_cost_witness"]
        selected_record = witness["play_route_record"]
        assert selected_record["route_id"] == witness["play_route_id"] == report["play_route_id"]
        assert (
            witness["play_route_signature"]
            == report["play_route_signature"]
            == selected_record["canonical_family_signature"]
            == ro.canonical_signature_from_actions(selected_record["actions"])
        )
        assert [int(row["event"]) for row in selected_record["actions"]] == [6, 7, 8]
        play_hit_rows = [
            row for row in selected_record["actions"] if int(row["hit_points"]) != 0
        ]
        save_actions = witness["save_route_actions"]
        assert [int(row["event"]) for row in save_actions] == [5, 6, 7, 8]
        save_hit_rows = [row for row in save_actions if int(row["hit_points"]) != 0]
        for role, expected in expected_costs.items():
            rows = play_hit_rows if role == "PLAY" else save_hit_rows
            event6_cost = next(
                (int(row["hit_points"]) for row in rows if int(row["event"]) == 6), 0,
            )
            assert event6_cost == expected
            if expected == 0:
                assert not rows
            phrase = f"FH {role} GW6 has source-replayed transfer hit cost 4"
            assert (phrase in message) is (expected == 4)
        # Reseal a false-zero route claim, then ask the real full verifier to
        # replay it again. Transfer costs are recomputed from the accepted
        # source state; changing the claim and its digests cannot erase them.
        import copy
        for role, expected in expected_costs.items():
            if expected == 0:
                continue
            forged = copy.deepcopy(gate_inputs[0])
            forged_report = forged["assembly_reports"][0]
            forged_witness = forged_report["hit_cost_witness"]
            if role == "PLAY":
                forged_record = forged_witness["play_route_record"]
                charged = next(
                    row for row in forged_record["actions"]
                    if int(row["hit_points"]) == 4
                )
                charged["hit_points"] = 0
                forged_witness["play_route_record_sha256"] = crf.canonical_sha256(
                    forged_record,
                )
            else:
                charged = next(
                    row for row in forged_witness["save_route_actions"]
                    if int(row["event"]) == 6 and int(row["hit_points"]) == 4
                )
                charged["hit_points"] = 0
            witness_body = dict(forged_witness)
            witness_body.pop("witness_sha256", None)
            forged_witness["witness_sha256"] = crf.canonical_sha256(witness_body)
            expected_error = "hit_points" if role == "PLAY" else "SAVE route witness differs"
            expected_error_type = (
                route_assembly.ChipRouteAssemblyError
                if role == "PLAY" else assembly.FutureStateAssemblyError
            )
            with pytest.raises(expected_error_type, match=expected_error):
                original_gate(conn, **forged)
        assert retained == []
        assert conn.total_changes == db_changes_before
        assert [int(row["event"]) for row in context["play_cache_loader_calls"]] == [6, 7, 8, 6, 7, 8]
        assert set(report["play_optimizer_world_info"]) == {"6", "7", "8"}
        assert all(
            int(report["play_optimizer_world_info"][str(event)]["union_players"])
            == len(context["play_optimizer_union_ids"])
            for event in (6, 7, 8)
        )

        evidence_root = context["evidence_root"]
        operation_id = evidence_ops._forecast_operation_id(
            evidence_ops._sha256(evidence_ops._origin_key(context["identity"])),
            "FH", observation_id,
        )
        metadata_root = evidence_ops._metadata_root(evidence_root)
        intent_path = metadata_root / f"forecast-intent-{operation_id}.json"
        publication_path = metadata_root / f"forecast-publication-{operation_id}.json"
        receipt_path = metadata_root / f"forecast-{operation_id}.json"
        assert intent_path.is_file() and not publication_path.exists() and not receipt_path.exists()
        intent_bytes = intent_path.read_bytes()
        artifact_bytes = {
            str(path.relative_to(evidence_root)): path.read_bytes()
            for path in evidence_root.rglob("*") if path.is_file()
        }
        maturation_lock_path = metadata_root / (
            f"maturation-{evidence_ops._maturation_key(observation_id, 5)}.lock"
        )
        maturation_lock_rel = str(maturation_lock_path.relative_to(evidence_root))
        assert maturation_lock_rel not in artifact_bytes
        with monkeypatch.context() as scoped:
            scoped.setattr(
                assembly, "build_future_free_hit_request",
                lambda *_args, **_kwargs: pytest.fail("incomplete forecast retry recomposed FH"),
            )
            with pytest.raises(
                evidence_ops.ChipEvidenceError,
                match="immutable intent but no complete publication",
            ):
                horizon_ops.build_future_action_forecast(
                    conn, identity=context["identity"], action="FH",
                    observation_id=observation_id, entry_id=241392,
                    continuation_generation_id=context["continuation"].generation_id,
                    value_generation_ids={}, expiry_event=5, coverage_product=context["coverage"],
                    evidence_root=evidence_root, cache_dir=context["cache_dir"],
                )
        assert intent_path.read_bytes() == intent_bytes
        assert {
            str(path.relative_to(evidence_root)): path.read_bytes()
            for path in evidence_root.rglob("*") if path.is_file()
        } == artifact_bytes
        assert conn.total_changes == db_changes_before

        all_players = [
            int(row["id"]) for row in conn.execute("SELECT id FROM players ORDER BY id")
        ]
        final_run = _archive_future_official_state(
            conn, context["raw_root"], (5, 6, 7, 8), final_events=(5,),
            live_players_by_event={5: all_players},
        )
        capture_changes_before = conn.total_changes
        captures_before = int(conn.execute(
            "SELECT COUNT(*) FROM outcome_observation_captures",
        ).fetchone()[0])
        with pytest.raises(evidence_ops.ChipEvidenceError, match="forecast"):
            evidence_ops.capture_outcome(
                conn, db_path=tmp_path / "origin.db", entry_id=241392,
                generation_id=context["generation"].generation_id,
                decision_id=context["decision_id"], route_id=context["route_id"],
                planning_event=4, cutoff=context["generation"].cutoff,
                action="FH", observation_id=observation_id, realization_event=5,
                fetch_run_id=final_run["run_id"], raw_root=context["raw_root"],
                evidence_root=evidence_root,
            )
        assert conn.total_changes == capture_changes_before
        assert int(conn.execute(
            "SELECT COUNT(*) FROM outcome_observation_captures",
        ).fetchone()[0]) == captures_before
        assert {
            str(path.relative_to(evidence_root)): path.read_bytes()
            for path in evidence_root.rglob("*") if path.is_file()
        } == artifact_bytes

        finalizer_calls = []
        with monkeypatch.context() as scoped:
            scoped.setattr(
                cal, "finalize_causal_observation",
                lambda *args, **kwargs: finalizer_calls.append((args, kwargs)),
            )
            with pytest.raises(evidence_ops.ChipEvidenceError, match="forecast"):
                evidence_ops.mature_observation(
                    conn, db_path=tmp_path / "origin.db", entry_id=241392,
                    generation_id=context["generation"].generation_id,
                    decision_id=context["decision_id"], route_id=context["route_id"],
                    planning_event=4, cutoff=context["generation"].cutoff,
                    action="FH", observation_id=observation_id,
                    realization_event=5, evidence_root=evidence_root,
                )
        assert not finalizer_calls
        assert conn.total_changes == capture_changes_before
        after_artifacts = {
            str(path.relative_to(evidence_root)): path.read_bytes()
            for path in evidence_root.rglob("*") if path.is_file()
        }
        assert maturation_lock_rel in after_artifacts
        assert after_artifacts[maturation_lock_rel] == b"\x00"
        assert {
            path: payload for path, payload in after_artifacts.items()
            if path != maturation_lock_rel
        } == artifact_bytes
    finally:
        conn.close()


def test_full_fh_source_replay_excludes_a_paid_pre_opportunity_event(
    tmp_path, monkeypatch,
):
    from fpl_brain import chip_horizon_evidence_assembly as assembly
    from fpl_brain import chip_evidence_ops as evidence_ops
    from fpl_brain import chip_horizon_evidence_ops as horizon_ops
    from fpl_brain import chip_route_assembly as cra, monte_carlo

    context = _prepare_public_fh_hit_gate_case(
        tmp_path, monkeypatch, historical_paid_hit_event=4,
        suppress_fh_play_standouts=True,
    )
    conn = context["conn"]
    try:
        source_route = cra.load_verified_normal_route(
            conn, context["decision_id"], route_id=context["route_id"],
            require_recommended=False,
        )
        old_action = source_route.partial.actions[0]
        assert int(old_action["event"]) == 4
        assert int(old_action["transition"].hit_points) == 4
        gate_results = []
        original_gate = assembly.verify_future_action_hit_cost_witnesses

        def capture_full_gate(conn, **kwargs):
            import inspect
            caller = inspect.currentframe().f_back
            try:
                module = caller.f_globals.get("__name__")
                name = caller.f_code.co_name
                line = caller.f_lineno
                assert module == "fpl_brain.chip_horizon_evidence_ops"
                if name == "build_future_action_forecast" and line == 852:
                    stage = "pre_retention_verification"
                elif name == "_forecast_receipt_from_publication" and line == 402:
                    stage = "publication_receipt_reverification"
                else:
                    raise AssertionError(
                        f"unexpected verifier delegation site: {module}.{name}:{line}"
                    )
                assert tuple(kwargs["expected_events"]) == (5,)
                proof = original_gate(conn, **kwargs)
                gate_results.append({
                    "stage": stage,
                    "proof_sha256": proof,
                    "delegated_to_original_verifier": True,
                    "expected_events": tuple(kwargs["expected_events"]),
                })
                return proof
            finally:
                del caller

        monkeypatch.setattr(assembly, "verify_future_action_hit_cost_witnesses", capture_full_gate)
        monkeypatch.setattr(
            monte_carlo, "simulate",
            lambda *_args, **_kwargs: pytest.fail("cache-only FH composition attempted world construction"),
        )
        forecast = horizon_ops.build_future_action_forecast(
            conn, identity=context["identity"], action="FH",
            observation_id="fh-history-before-opportunity", entry_id=241392,
            continuation_generation_id=context["continuation"].generation_id,
            value_generation_ids={}, expiry_event=5, coverage_product=context["coverage"],
            evidence_root=context["evidence_root"], cache_dir=context["cache_dir"],
        )
        assert [row["stage"] for row in gate_results] == [
            "pre_retention_verification",
            "publication_receipt_reverification",
        ]
        assert all(row["delegated_to_original_verifier"] is True for row in gate_results)
        assert [row["expected_events"] for row in gate_results] == [(5,), (5,)]
        assert gate_results[0]["proof_sha256"] == gate_results[1]["proof_sha256"]
        assert forecast["receipt"]["hit_cost_proof_sha256"] == gate_results[1]["proof_sha256"]
        assert forecast["forecast"]["covered_events"] == [5]
        report = forecast["assembly_reports"][0]
        assert [int(row["event"]) for row in context["play_cache_loader_calls"]] == [6, 7, 8, 6, 7, 8]
        assert set(report["play_optimizer_world_info"]) == {"6", "7", "8"}
        assert all(
            int(report["play_optimizer_world_info"][str(event)]["union_players"])
            == len(context["play_optimizer_union_ids"])
            for event in (6, 7, 8)
        )
        witness = report["hit_cost_witness"]
        assert [int(row["event"]) for row in witness["save_route_actions"]] == [5, 6, 7, 8]
        assert all(int(row["hit_points"]) == 0 for row in witness["save_route_actions"])
        assert [int(row["event"]) for row in witness["play_route_record"]["actions"]] == [6, 7, 8]
    finally:
        conn.close()
