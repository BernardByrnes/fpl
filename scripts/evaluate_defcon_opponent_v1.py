"""Evaluate the DEFCON model against the Brain baseline on realised official data.

READ-ONLY with respect to the live database.  This driver never opens fpl.db for
writing, never runs a production model, and never mutates the external cache.

Unit of analysis is the PLAYER-MATCH.  FPL awards DEFCON points for reaching the
action threshold in a SINGLE match, so a double gameweek is two independent
threshold crossings; players with more than one match in a gameweek are excluded
rather than summed.

EXPOSURE IS ORACLE.  ``expected_minutes`` is the realised minutes.  That is
deliberate: it isolates the count/tail model from the minutes model so an
improvement here cannot be minutes-model luck.  Production conditions the same
probability on PREDICTED minutes, so these numbers are an upper bound on
end-to-end accuracy and are reported as such.

TARGET PROVENANCE.  The label is the official FPL ``defensive_contribution``.
For 2026/27 it is ALSO present in the Brain's own database, so the external
carrier is validated against official Brain-ingested values before any 2025/26
fold is believed (see ``cross_check_against_brain``).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import defcon_model as dm  # noqa: E402
from fpl_brain import scoring_rules as sr  # noqa: E402

EXTERNAL_ROOT = Path(
    "K:/FPL-core/data/exports/fpl_core/extracted/"
    "FPL-Core-Insights-d2eee2a25645ec4a73bda640f0d92048166fa22c/data"
)
LIVE_DB = "K:/FPL/fpl.db"
OUT = Path("K:/FPL-pred/data/exports/predictive_v1")
RULES = sr.DEFAULT_SCORING_RULES

# --- MATERIALITY BAR (declared BEFORE any result was inspected) --------------
# A DEFCON model change is MATERIAL only if ALL hold on the pooled rolling OOS
# evaluation:
#   1. pooled Brier improves by >= 0.0050 ABSOLUTE.  DEFCON is a 2-point
#      threshold crossing; a sub-0.005 probability move shifts expected points by
#      under 0.01, below the resolution at which it could change a real lineup.
#   2. the improvement holds in >= 60% of folds (stability, not one lucky fold);
#   3. pooled expected-points bias does not get worse.
# Otherwise: WATCH if pooled Brier improves at all, else REJECT.
MATERIAL_BRIER_ABS_IMPROVEMENT = 0.0050
MATERIAL_FOLD_FRACTION = 0.60


def _f(value, default=0.0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pid(row: dict) -> int:
    """Player key: ``player_gameweek_stats`` uses ``id``, the others ``player_id``."""

    return int(_f(row.get("player_id") or row.get("id")))


#: The external roster spells positions out; the Brain's scoring rules are keyed
#: on the FPL element codes.  Both spellings are accepted so a schema drift in
#: either direction cannot silently empty the sample.
POSITION_CODES = {
    "goalkeeper": "GKP", "keeper": "GKP", "gkp": "GKP", "gk": "GKP",
    "defender": "DEF", "def": "DEF",
    "midfielder": "MID", "mid": "MID",
    "forward": "FWD", "fwd": "FWD", "striker": "FWD",
}


def _position_code(value: object) -> str:
    return POSITION_CODES.get(str(value or "").strip().lower(), "")


def load_season(season: str) -> list[dict]:
    """Player-match rows for one season, Premier League tournament only."""

    base = EXTERNAL_ROOT / season / "By Tournament" / "Premier League"
    if not base.exists():
        raise FileNotFoundError(base)

    rows: list[dict] = []
    for gw_dir in sorted(base.glob("GW*"), key=lambda p: int(p.name[2:])):
        gw = int(gw_dir.name[2:])
        stats = _read(gw_dir / "player_gameweek_stats.csv")
        player_matches = _read(gw_dir / "playermatchstats.csv")
        matches = _read(gw_dir / "matches.csv")
        roster = _read(gw_dir / "players.csv")
        teams = _read(gw_dir / "teams.csv")
        if not (stats and player_matches and matches and roster and teams):
            continue

        code_to_team_id = {str(t.get("code")): int(_f(t.get("id"))) for t in teams if _f(t.get("id"))}
        position = {_pid(r): _position_code(r.get("position")) for r in roster if _pid(r)}
        team_of = {
            _pid(r): code_to_team_id.get(str(r.get("team_code")))
            for r in roster
            if _pid(r)
        }
        match_info = {r["match_id"]: r for r in matches}

        per_player: dict[int, list[dict]] = defaultdict(list)
        for r in player_matches:
            pid = _pid(r)
            if pid:
                per_player[pid].append(r)

        for r in stats:
            pid = _pid(r)
            minutes = _f(r.get("minutes"))
            if not pid or minutes <= 0:
                continue
            played = [m for m in per_player.get(pid, []) if _f(m.get("minutes_played")) > 0]
            if len(played) != 1:
                continue
            info = match_info.get(played[0]["match_id"])
            team_id = team_of.get(pid)
            if info is None or team_id is None:
                continue
            pos = position.get(pid, "")
            threshold = RULES.defcon_threshold_for(pos)
            if threshold is None or pos not in RULES.defcon_positions:
                continue

            is_home = int(_f(info.get("home_team"))) == team_id
            opp_prefix = "away" if is_home else "home"
            actions = _f(r.get("defensive_contribution"))
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "player_id": pid,
                    "position": pos,
                    "is_home": is_home,
                    "minutes": minutes,
                    "actions": actions,
                    "threshold": int(threshold),
                    "hit": int(actions >= int(threshold)),
                    "opp_possession": _f(info.get(f"{opp_prefix}_possession")),
                    "opp_box_shots": _f(info.get(f"{opp_prefix}_shots_inside_box")),
                    "opp_shots": _f(info.get(f"{opp_prefix}_total_shots")),
                    "opp_xg": _f(info.get(f"{opp_prefix}_expected_goals_xg")),
                    "xg": _f(r.get("expected_goals")),
                    "xa": _f(r.get("expected_assists")),
                    "points": _f(r.get("total_points")),
                }
            )
    return rows


def _pressure(row: dict) -> float:
    """Opponent attacking pressure: possession share plus box-shot volume."""

    return row["opp_possession"] / 100.0 + row["opp_box_shots"] / 10.0


def _own_rates(prior: list[dict]) -> dict[int, float]:
    """Shrunk per-90 defensive-action rate from PRIOR matches only."""

    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in prior:
        totals[r["player_id"]][0] += r["actions"]
        totals[r["player_id"]][1] += r["minutes"]
    pooled: dict[str, list[float]] = defaultdict(list)
    for r in prior:
        pooled[r["position"]].append(r["actions"] * 90.0 / max(1.0, r["minutes"]))
    pooled_rate = {pos: (sum(v) / len(v) if v else 0.0) for pos, v in pooled.items()}
    return {
        pid: dm.defcon_rate_posterior(
            observed_actions=acts,
            observed_minutes=mins,
            prior_rate_per90=pooled_rate.get(pos, 0.0),
            prior_ess_minutes=600.0,
        )
        for pid, (acts, mins) in totals.items()
        for pos in [_position_of(prior, pid)]
    }


def _position_of(prior: list[dict], player_id: int) -> str:
    for r in prior:
        if r["player_id"] == player_id:
            return r["position"]
    return ""


def fit_opponent_beta(prior: list[dict], own_rates: dict[int, float]) -> dict:
    """One-covariate Poisson GLM for the opponent-pressure multiplier.

    log(mu_i) = log(exposure_i) + log(rate_i) + beta * z_i.  beta solves the
    Poisson score equation sum(z_i (y_i - mu_i)) = 0 on PRIOR data by bisection.
    """

    if len(prior) < 200:
        return {"beta": 0.0, "z_mean": 0.0, "z_sd": 1.0, "n": len(prior)}
    raw = [_pressure(r) for r in prior]
    z_mean = sum(raw) / len(raw)
    z_sd = statistics.pstdev(raw) or 1.0
    z = [(p - z_mean) / z_sd for p in raw]
    exposures = [max(0.0, r["minutes"]) / 90.0 for r in prior]
    rates = [max(1e-6, own_rates.get(r["player_id"], 0.0)) for r in prior]
    counts = [r["actions"] for r in prior]

    def score(beta: float) -> float:
        return sum(
            zi * (y - exposure * rate * math.exp(beta * zi))
            for zi, exposure, rate, y in zip(z, exposures, rates, counts)
        )

    lo, hi = -3.0, 3.0
    s_lo, s_hi = score(lo), score(hi)
    beta = 0.0
    if s_lo * s_hi < 0:
        for _ in range(80):
            mid = (lo + hi) / 2
            s_mid = score(mid)
            if s_lo * s_mid <= 0:
                hi, s_hi = mid, s_mid
            else:
                lo, s_lo = mid, s_mid
        beta = (lo + hi) / 2
    return {"beta": beta, "z_mean": z_mean, "z_sd": z_sd, "n": len(prior)}



def _prior_means(prior, own_rates, stats, fit, variant, prior_rate):
    """Per-row ``(mean_actions_at_scale_1, threshold, hit)`` for the fitting set.

    The level scale multiplies the mean, so it does not change the threshold or
    the dispersion.  Precomputing the unscaled means makes each candidate scale a
    pure tail evaluation instead of a full projection rebuild, which is what
    keeps the fit affordable.
    """

    rows = []
    for r in prior:
        rate = own_rates.get(r["player_id"], prior_rate.get(r["position"], 0.0))
        z = (_pressure(r) - fit["z_mean"]) / (fit["z_sd"] or 1.0)
        base = dm.project_defcon(
            position=r["position"], expected_minutes=r["minutes"], rate_posterior=rate,
            variant=variant, dispersion_r=dm.dispersion_for(stats, r["position"]),
            opponent_multiplier=math.exp(fit["beta"] * z), level_scale=1.0,
        )
        if base.threshold is None:
            continue
        rows.append((base.projected_defcon, base.threshold, r["hit"], base.dispersion_r))
    return rows


def _tail_for(variant: str, mean: float, threshold: int, dispersion_r: float) -> float:
    if variant in (dm.VARIANT_LEVEL_CALIBRATED_POISSON,):
        return dm.xpts.poisson_tail_probability(mean, threshold)
    return dm.negative_binomial_tail(mean, dispersion_r, threshold)


def fit_level_scale(prior, own_rates, stats, fit, variant, *, objective="brier"):
    """Fit one multiplicative rate correction on PRIOR data only.

    ``objective='brier'`` minimises the prior Brier score, the metric the fold is
    judged on.  ``objective='mean'`` instead matches the mean predicted
    probability to the observed hit rate, which is the correction that removes
    expected-POINTS bias.  They are reported separately because Brier is
    dominated by the large low-probability mass while expected points weight
    every row equally.

    Fitted on prior folds and applied unchanged to the scored fold, so it can
    never see the data it is judged on.
    """

    if not prior:
        return 1.0
    sample = prior if len(prior) <= 3000 else prior[:: max(1, len(prior) // 3000)]
    pooled_by_position = defaultdict(list)
    for r in prior:
        pooled_by_position[r["position"]].append(r["actions"] * 90.0 / max(1.0, r["minutes"]))
    prior_rate = {pos: sum(v) / len(v) for pos, v in pooled_by_position.items() if v}

    prepared = _prior_means(sample, own_rates, stats, fit, variant, prior_rate)
    if not prepared:
        return 1.0
    observed = sum(hit for _, _, hit, _ in prepared) / len(prepared)

    def objective_value(scale: float) -> float:
        if objective == "brier":
            total = 0.0
            for mean, threshold, hit, r in prepared:
                total += (_tail_for(variant, mean * scale, threshold, r) - hit) ** 2
            return total / len(prepared)
        total = 0.0
        for mean, threshold, _, r in prepared:
            total += _tail_for(variant, mean * scale, threshold, r)
        return abs(total / len(prepared) - observed)

    grid = [0.05 * (1.3 ** k) for k in range(0, 26)]
    best = min(grid, key=objective_value)
    lo, hi = best / 1.3, best * 1.3
    for _ in range(30):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if objective_value(m1) < objective_value(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def evaluate(rows: list[dict], *, min_origin: int = 8, label: str = "") -> dict:
    pooled: dict[str, dict[str, list]] = {}
    fold_beta: dict[int, float] = {}
    fold_scale: dict[int, dict[str, float]] = {}

    for origin in sorted({r["gw"] for r in rows if r["gw"] >= min_origin}):
        prior = [r for r in rows if r["gw"] < origin]
        test = [r for r in rows if r["gw"] == origin]
        if len(prior) < 300 or not test:
            continue

        own_rates = _own_rates(prior)
        stats = dm.fit_count_statistics(prior)
        fit = fit_opponent_beta(prior, own_rates)
        fold_beta[origin] = round(fit["beta"], 6)
        pooled_rate: dict[str, list[float]] = defaultdict(list)
        for r in prior:
            pooled_rate[r["position"]].append(r["actions"] * 90.0 / max(1.0, r["minutes"]))
        prior_rate = {pos: sum(v) / len(v) for pos, v in pooled_rate.items() if v}

        level_variants = (dm.VARIANT_LEVEL_CALIBRATED_POISSON, dm.VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL)
        level_scales = {
            variant: fit_level_scale(prior, own_rates, stats, fit, variant, objective="brier")
            for variant in level_variants
        }
        mean_scales = {
            variant: fit_level_scale(prior, own_rates, stats, fit, variant, objective="mean")
            for variant in level_variants
        }
        fold_scale[origin] = {f"{k}_brier": round(v, 6) for k, v in level_scales.items()}
        fold_scale[origin].update({f"{k}_mean": round(v, 6) for k, v in mean_scales.items()})

        per_variant: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for r in test:
            rate = own_rates.get(
                r["player_id"],
                prior_rate.get(r["position"], 0.0),
            )
            z = (_pressure(r) - fit["z_mean"]) / (fit["z_sd"] or 1.0)
            multiplier = math.exp(fit["beta"] * z)
            dispersion = dm.dispersion_for(stats, r["position"])
            for variant in dm.DEFCON_VARIANTS:
                projection = dm.project_defcon(
                    position=r["position"],
                    expected_minutes=r["minutes"],
                    rate_posterior=rate,
                    variant=variant,
                    dispersion_r=dispersion,
                    opponent_multiplier=multiplier,
                    level_scale=level_scales.get(variant, 1.0),
                    player_id=r["player_id"],
                )
                per_variant[variant].append((projection.p_threshold, r["hit"]))

        for variant, pairs in per_variant.items():
            bucket = pooled.setdefault(variant, {"p": [], "y": [], "fold": []})
            for p, y in pairs:
                bucket["p"].append(p)
                bucket["y"].append(y)
                bucket["fold"].append(origin)

    base = pooled.get(dm.VARIANT_BASELINE_POISSON, {"p": [], "y": [], "fold": []})
    results: dict[str, dict] = {}
    for variant, bucket in pooled.items():
        if not bucket["p"]:
            continue
        per_fold = []
        improved = 0
        for fold in sorted(set(bucket["fold"])):
            idx = [i for i, f in enumerate(bucket["fold"]) if f == fold]
            y_f = [bucket["y"][i] for i in idx]
            b_f = dm.brier_score([bucket["p"][i] for i in idx], y_f)
            b_base = dm.brier_score([base["p"][i] for i in idx], y_f)
            improved += int(b_f < b_base)
            per_fold.append({"fold": fold, "n": len(idx), "brier": b_f, "baseline_brier": b_base, "delta": b_f - b_base})
        results[variant] = {
            "n": len(bucket["p"]),
            "observed_hit_rate": sum(bucket["y"]) / len(bucket["y"]),
            "brier": dm.brier_score(bucket["p"], bucket["y"]),
            "log_loss": dm.log_loss(bucket["p"], bucket["y"]),
            "expected_points": dm.expected_points_calibration(bucket["p"], bucket["y"]),
            "calibration": dm.calibration_table(bucket["p"], bucket["y"]),
            "folds": len(per_fold),
            "folds_improving_vs_baseline": improved,
            "fold_fraction_improving": improved / len(per_fold) if per_fold else 0.0,
            "per_fold": per_fold,
        }
    return {
        "label": label,
        "rows": len(rows),
        "fold_count": len(fold_beta),
        "opponent_beta_by_fold": fold_beta,
        "level_scale_by_fold": fold_scale,
        "materiality_bar": {
            "brier_abs_improvement": MATERIAL_BRIER_ABS_IMPROVEMENT,
            "fold_fraction": MATERIAL_FOLD_FRACTION,
        },
        "variants": results,
    }


def verdict_for(result: dict) -> dict[str, str]:
    base = result["variants"].get(dm.VARIANT_BASELINE_POISSON)
    if not base:
        return {}
    out = {dm.VARIANT_BASELINE_POISSON: "BASELINE"}
    for variant, entry in result["variants"].items():
        if variant == dm.VARIANT_BASELINE_POISSON:
            continue
        delta = entry["brier"] - base["brier"]
        bias_ok = abs(entry["expected_points"]["signed_bias_points"]) <= abs(
            base["expected_points"]["signed_bias_points"]
        )
        if (
            -delta >= MATERIAL_BRIER_ABS_IMPROVEMENT
            and entry["fold_fraction_improving"] >= MATERIAL_FOLD_FRACTION
            and bias_ok
        ):
            out[variant] = "ACCEPT"
        elif delta < 0:
            out[variant] = "WATCH"
        else:
            out[variant] = "REJECT"
    return out


def cross_check_against_brain(season: str, rows: list[dict]) -> dict:
    """Prove the external carrier matches official Brain-ingested DEFCON.

    Only possible for the season present in the live database (2026/27).  If the
    external mirror disagrees with the Brain's own official values for the SAME
    (player, event), the label provenance for every other fold is unsound and
    the caller must treat the evaluation as unproven.
    """

    from fpl_brain.database import connect_readonly_database

    conn = connect_readonly_database(LIVE_DB)
    try:
        official = {
            (int(r["player_id"]), int(r["event"])): float(r["defensive_contribution"])
            for r in conn.execute(
                """SELECT pg.player_id, pg.event, pg.defensive_contribution
                   FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
                   WHERE f.finished = 1 AND f.started = 1 AND pg.defensive_contribution IS NOT NULL"""
            )
        }
    finally:
        conn.close()

    external = {(r["player_id"], r["gw"]): r["actions"] for r in rows if r["season"] == season}
    shared = sorted(set(official) & set(external))
    agree = [k for k in shared if abs(official[k] - external[k]) < 1e-9]
    return {
        "brain_official_rows": len(official),
        "external_rows": len(external),
        "shared_keys": len(shared),
        "exact_agreements": len(agree),
        "agreement_rate": (len(agree) / len(shared)) if shared else None,
        "disagreements": [
            {"player_id": k[0], "event": k[1], "brain": official[k], "external": external[k]}
            for k in shared
            if abs(official[k] - external[k]) >= 1e-9
        ][:10],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "materiality_bar": {
            "brier_abs_improvement": MATERIAL_BRIER_ABS_IMPROVEMENT,
            "fold_fraction": MATERIAL_FOLD_FRACTION,
            "declared": "before any result was inspected",
        },
        "exposure": "oracle (realised minutes); isolates the count model from the minutes model",
        "unit": "player-match; single-match gameweeks only",
    }

    for season in ("2026-2027", "2025-2026"):
        rows = load_season(season)
        print(f"{season}: {len(rows)} usable player-match rows across {len({r['gw'] for r in rows})} gameweeks")
        if len({r["gw"] for r in rows}) < 3:
            print(f"  too few gameweeks for folds; skipping")
            continue
        if season == "2026-2027":
            payload["cross_check"] = cross_check_against_brain(season, rows)
        result = evaluate(rows, label=season)
        result["verdicts"] = verdict_for(result)
        payload.setdefault("seasons", {})[season] = result

    (OUT / "defcon_evaluation.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    for season, result in payload.get("seasons", {}).items():
        print(f"\n=== {season} (folds={result['fold_count']}, n={result['rows']}) ===")
        base = result["variants"].get(dm.VARIANT_BASELINE_POISSON)
        if not base:
            print("  no folds produced (insufficient prior history); cross-check only")
            continue
        ep = base.get("expected_points", {})
        print(f"  baseline Brier {base['brier']:.6f} logloss {base['log_loss']:.6f} "
              f"pred {ep.get('mean_predicted_probability', float('nan')):.4f} "
              f"actual {ep.get('realised_hit_rate', float('nan')):.4f}")
        for variant, entry in result["variants"].items():
            print(f"  {variant:38s} Brier {entry['brier']:.6f} (d {entry['brier']-base['brier']:+.6f}) "
                  f"folds+ {entry['folds_improving_vs_baseline']}/{entry['folds']} -> {result['verdicts'].get(variant)}")
    if "cross_check" in payload:
        print("\n=== cross-check vs Brain official DEFCON (2026/27) ===")
        print(" ", json.dumps(payload["cross_check"], indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
