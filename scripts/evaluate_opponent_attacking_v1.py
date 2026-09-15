"""Opponent/team and attacking feature evaluation against official targets.

Two questions, both answered with point-in-time rolling origins over 2025/26:

TEAM/OPPONENT (§5).  The Brain already fits team attack/defence from official xG
and folds in home/away strength.  That multiplicative structure is reproduced
here as the BASELINE predictor of a side's goals, using official xG only.  Each
candidate then adds ONE external opponent-process measure, and we ask whether the
residual improves.  A candidate that merely restates team xGA cannot help, and
that is exactly what is being tested.

ATTACKING (§6).  Baseline is the Brain's own player-rate construction: the
player's prior official xG/90 (or xA/90) shrunk toward a position-pooled prior.
Candidates add external process rates.  The target is the player's REALISED
official xG/xA in the scored gameweek, so the label is official FPL, not a
third-party stand-in.

Materiality bars are declared before the results.
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
from collections import OrderedDict  # noqa: E402

EXTERNAL_ROOT = Path(
    "K:/FPL-core/data/exports/fpl_core/extracted/"
    "FPL-Core-Insights-d2eee2a25645ec4a73bda640f0d92048166fa22c/data"
)
OUT = Path("K:/FPL-pred/data/exports/predictive_v1")

# --- MATERIALITY BARS (declared BEFORE any result was inspected) ------------
# TEAM/OPPONENT: a candidate is MATERIAL only if it improves the pooled Poisson
# NLL by >= 0.0050 nats per side-match (a goal is worth ~0.1 nats of λ accuracy;
# sub-0.005 cannot move a clean-sheet or goal expectation enough to reorder a
# lineup) AND improves in >= 60% of folds.
TEAM_NLL_ABS_IMPROVEMENT = 0.0050
# ATTACKING: a candidate is MATERIAL only if it improves MAE by >= 1.0% RELATIVE
# on the realised official xG/xA AND improves in >= 60% of folds.
ATTACKING_REL_IMPROVEMENT = 0.010
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


def poisson_nll(lam: float, observed: int) -> float:
    lam = max(1e-9, float(lam))
    return lam - observed * math.log(lam) + math.lgamma(observed + 1)


def _sides(season: str) -> list[dict]:
    """One row per (match, side) with the side's own and opponent's official xG."""

    base = EXTERNAL_ROOT / season / "By Tournament" / "Premier League"
    rows = []
    for gw_dir in sorted(base.glob("GW*"), key=lambda p: int(p.name[2:])):
        gw = int(gw_dir.name[2:])
        for m in _read(gw_dir / "matches.csv"):
            for own, opp in (("home", "away"), ("away", "home")):
                if _f(m.get(f"{own}_score")) is None and m.get(f"{own}_score") in (None, ""):
                    continue
                rows.append(
                    {
                        "gw": gw,
                        "match_id": m.get("match_id"),
                        "side": own,
                        "team": int(_f(m.get(f"{own}_team"))),
                        "opponent": int(_f(m.get(f"{opp}_team"))),
                        "goals": _f(m.get(f"{own}_score")),
                        "net_goals": _f(m.get(f"{own}_score")) - _f(m.get(f"{opp}_score")),
                        "xg_for": _f(m.get(f"{own}_expected_goals_xg")),
                        "xg_against": _f(m.get(f"{opp}_expected_goals_xg")),
                        "opp_box_shots_allowed": _f(m.get(f"{opp}_shots_inside_box")),
                        "opp_shots_allowed": _f(m.get(f"{opp}_total_shots")),
                        "opp_sot_allowed": _f(m.get(f"{opp}_shots_on_target")),
                        "opp_npxg_allowed": _f(m.get(f"{opp}_non_penalty_xg")),
                        "opp_big_chances_allowed": _f(m.get(f"{opp}_big_chances")),
                        "opp_box_touches_allowed": _f(m.get(f"{opp}_touches_in_opposition_box")),
                        "opp_possession": _f(m.get(f"{opp}_possession")),
                        "opp_xgot_allowed": _f(m.get(f"{opp}_xg_on_target_xgot")),
                    }
                )
    return [r for r in rows if r["team"] and r["opponent"]]


def evaluate_team(rows: list[dict], *, min_origin: int = 8) -> dict:
    candidates = (
        "opp_box_shots_allowed", "opp_shots_allowed", "opp_sot_allowed",
        "opp_npxg_allowed", "opp_big_chances_allowed", "opp_box_touches_allowed",
        "opp_xgot_allowed",
    )
    gws = sorted({r["gw"] for r in rows})
    pooled: dict[str, list] = {"BASELINE": [], "CONSTANT": []}
    for name in candidates:
        pooled[name] = []
    fold_records: dict[str, list[float]] = defaultdict(list)

    for origin in [g for g in gws if g >= min_origin]:
        prior = [r for r in rows if r["gw"] < origin]
        test = [r for r in rows if r["gw"] == origin]
        if len(prior) < 200 or not test:
            continue
        league_avg = max(1e-6, sum(r["goals"] for r in prior) / len(prior))

        def strength(rows_, key):
            by_team: dict[int, list[float]] = defaultdict(list)
            for r in rows_:
                by_team[r["team"]].append(r[key])
            return {t: (sum(v) / len(v)) for t, v in by_team.items() if v}

        attack_xg = strength(prior, "xg_for")
        defence_xg = strength(prior, "xg_against")
        # Candidate covariates: the opponent's per-match allowance, shrunk to the
        # league mean with a 6-match equivalent prior.
        cov = {}
        for name in candidates:
            by_team: dict[int, list[float]] = defaultdict(list)
            for r in prior:
                by_team[r["team"]].append(r[name])
            cov[name] = {
                t: (sum(v) + 6.0 * (sum(x[name] for x in prior) / len(prior))) / (len(v) + 6.0)
                for t, v in by_team.items()
            }
        cov_league = {n: (sum(r[n] for r in prior) / len(prior)) for n in candidates}

        fold_nll: dict[str, float] = {}
        fold_nll["CONSTANT"] = sum(
            poisson_nll(league_avg, int(r["goals"])) for r in test
        ) / len(test)
        for name in ["BASELINE"] + list(candidates):
            total = 0.0
            for r in test:
                lam = league_avg
                lam *= max(1e-3, attack_xg.get(r["team"], league_avg)) / league_avg
                if name == "BASELINE":
                    lam *= max(1e-3, defence_xg.get(r["opponent"], league_avg)) / league_avg
                else:
                    ratio = cov[name].get(r["opponent"], cov_league[name]) / max(1e-6, cov_league[name])
                    lam *= max(1e-3, defence_xg.get(r["opponent"], league_avg)) / league_avg
                    lam *= max(0.2, min(5.0, ratio))
                total += poisson_nll(lam, int(r["goals"]))
            fold_nll[name] = total / len(test)
        for name, value in fold_nll.items():
            pooled[name].append(value)
            fold_records[name].append(value - fold_nll["BASELINE"])

    base_nll = sum(pooled["BASELINE"]) / len(pooled["BASELINE"])
    out = {}
    for name, values in pooled.items():
        mean_nll = sum(values) / len(values)
        deltas = fold_records[name] if name != "BASELINE" else [0.0] * len(values)
        improved = sum(1 for d in deltas if d < 0)
        delta = mean_nll - base_nll
        if name in ("BASELINE", "CONSTANT"):
            verdict = name
        elif -delta >= TEAM_NLL_ABS_IMPROVEMENT and improved / len(deltas) >= MATERIAL_FOLD_FRACTION:
            verdict = "ACCEPT"
        elif delta < 0:
            verdict = "WATCH"
        else:
            verdict = "REJECT"
        out[name] = {
            "poisson_nll": mean_nll,
            "delta_vs_baseline": delta,
            "folds": len(values),
            "folds_improving": improved,
            "fold_fraction_improving": improved / len(values) if values else 0.0,
            "verdict": verdict,
        }
    return {"baseline_nll": base_nll, "candidates": out, "folds": len(pooled["BASELINE"])}


def _ols(xs: list[list[float]], ys: list[float]) -> list[float]:
    """Least squares with an intercept, by normal equations (2 or 3 predictors)."""

    k = len(xs[0]) + 1
    design = [[1.0] + row for row in xs]
    ata = [[sum(design[i][a] * design[i][b] for i in range(len(design))) for b in range(k)] for a in range(k)]
    aty = [sum(design[i][a] * ys[i] for i in range(len(ys))) for a in range(k)]
    # Gaussian elimination with partial pivoting
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(ata[r][col]))
        ata[col], ata[pivot] = ata[pivot], ata[col]
        aty[col], aty[pivot] = aty[pivot], aty[col]
        if abs(ata[col][col]) < 1e-12:
            continue
        for r in range(k):
            if r == col:
                continue
            factor = ata[r][col] / ata[col][col]
            for c in range(col, k):
                ata[r][c] -= factor * ata[col][c]
            aty[r] -= factor * aty[col]
    return [aty[i] / ata[i][i] if abs(ata[i][i]) > 1e-12 else 0.0 for i in range(k)]


def evaluate_attacking(season: str, *, min_origin: int = 8) -> dict:
    """Incremental value of a process rate for predicting the SAME quantity.

    Both arms are fitted on PRIOR folds and scored on the next fold, so the
    candidate cannot win by over-fitting the fold it is judged on:

      BASELINE   next target ~ shrunk own prior rate (the Brain construction)
      CANDIDATE  next target ~ BASELINE + the external process rate

    MAE is computed on identical rows, so the comparison is paired.
    """

    base = EXTERNAL_ROOT / season / "By Tournament" / "Premier League"
    obs: list[dict] = []
    for gw_dir in sorted(base.glob("GW*"), key=lambda p: int(p.name[2:])):
        gw = int(gw_dir.name[2:])
        stats = _read(gw_dir / "player_gameweek_stats.csv")
        pms = {int(_f(r["player_id"])): r for r in _read(gw_dir / "playermatchstats.csv") if _f(r["player_id"])}
        for r in stats:
            pid = int(_f(r.get("player_id") or r.get("id")))
            minutes = _f(r.get("minutes"))
            if not pid or minutes <= 0:
                continue
            pm = pms.get(pid, {})
            obs.append({
                "gw": gw, "player_id": pid, "minutes": minutes,
                "xg": _f(r.get("expected_goals")), "xa": _f(r.get("expected_assists")),
                "shots": _f(pm.get("total_shots")), "sot": _f(pm.get("shots_on_target")),
                "chances_created": _f(pm.get("chances_created")),
                "box_touches": _f(pm.get("touches_opposition_box")),
            })

    gws = sorted({r["gw"] for r in obs})
    results: dict[str, dict] = {}
    for target, candidate in (("xg", "shots"), ("xg", "sot"), ("xg", "box_touches"),
                              ("xa", "chances_created")):
        fold_deltas: list[float] = []
        base_maes: list[float] = []
        cand_maes: list[float] = []
        for origin in [g for g in gws if g >= min_origin]:
            prior = [r for r in obs if r["gw"] < origin]
            test = [r for r in obs if r["gw"] == origin]
            if len(prior) < 400 or len(test) < 50:
                continue
            pooled_target = sum(r[target] for r in prior) / sum(r["minutes"] for r in prior) * 90.0
            pooled_cand = sum(r[candidate] for r in prior) / sum(r["minutes"] for r in prior) * 90.0

            def features(rows_):
                own_t: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
                own_c: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
                for r in rows_:
                    own_t[r["player_id"]][0] += r[target]
                    own_t[r["player_id"]][1] += r["minutes"]
                    own_c[r["player_id"]][0] += r[candidate]
                out = []
                for r in rows_:
                    t, m = own_t[r["player_id"]]
                    c, _ = own_c[r["player_id"]]
                    base_rate = dm.defcon_rate_posterior(
                        observed_actions=t, observed_minutes=m,
                        prior_rate_per90=pooled_target, prior_ess_minutes=850.0)
                    cand_rate = dm.defcon_rate_posterior(
                        observed_actions=c, observed_minutes=m,
                        prior_rate_per90=pooled_cand, prior_ess_minutes=850.0)
                    out.append((base_rate, cand_rate, r["minutes"]))
                return out

            train = features(prior)
            fit_base = _ols([[b * mn / 90.0] for b, _, mn in train], [r[target] for r in prior])
            fit_cand = _ols([[b * mn / 90.0, c * mn / 90.0] for b, c, mn in train], [r[target] for r in prior])
            test_rows = features(test)
            y = [r[target] for r in test]
            b_err = [abs(fit_base[0] + fit_base[1] * bx - yy) for (bx, _, _), yy in zip(test_rows, y)]
            c_err = [abs(fit_cand[0] + fit_cand[1] * bx + fit_cand[2] * cx - yy)
                     for (bx, cx, _), yy in zip(test_rows, y)]
            bm, cm = sum(b_err) / len(b_err), sum(c_err) / len(c_err)
            base_maes.append(bm)
            cand_maes.append(cm)
            fold_deltas.append(cm - bm)

        if not base_maes:
            continue
        base_mae = sum(base_maes) / len(base_maes)
        cand_mae = sum(cand_maes) / len(cand_maes)
        improved = sum(1 for d in fold_deltas if d < 0)
        rel = (base_mae - cand_mae) / base_mae if base_mae else 0.0
        if rel >= ATTACKING_REL_IMPROVEMENT and improved / len(fold_deltas) >= MATERIAL_FOLD_FRACTION:
            verdict = "ACCEPT"
        elif cand_mae < base_mae:
            verdict = "WATCH"
        else:
            verdict = "REJECT"
        results[f"{candidate}_per90 -> next {target}"] = {
            "baseline_mae": base_mae, "candidate_mae": cand_mae,
            "relative_improvement": rel, "folds": len(fold_deltas),
            "folds_improving": improved,
            "fold_fraction_improving": improved / len(fold_deltas),
            "verdict": verdict,
        }
    return results


def double_counting(season: str) -> dict:
    """Pearson r between each external process measure and the official quantity
    it would restate, on the same player-match rows."""

    base = EXTERNAL_ROOT / season / "By Tournament" / "Premier League"
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for gw_dir in sorted(base.glob("GW*"), key=lambda p: int(p.name[2:])):
        stats = _read(gw_dir / "player_gameweek_stats.csv")
        pms = {int(_f(r["player_id"])): r for r in _read(gw_dir / "playermatchstats.csv") if _f(r["player_id"])}
        for r in stats:
            pid = int(_f(r.get("player_id") or r.get("id")))
            minutes = _f(r.get("minutes"))
            if not pid or minutes <= 0 or pid not in pms:
                continue
            pm = pms[pid]
            pairs["shots vs official xG"].append((_f(pm.get("total_shots")), _f(r.get("expected_goals"))))
            pairs["SOT vs official xG"].append((_f(pm.get("shots_on_target")), _f(r.get("expected_goals"))))
            pairs["chances_created vs official xA"].append((_f(pm.get("chances_created")), _f(r.get("expected_assists"))))
            pairs["box_touches vs official xG"].append((_f(pm.get("touches_opposition_box")), _f(r.get("expected_goals"))))
            pairs["defensive actions vs official DEFCON"].append(
                (
                    _f(pm.get("tackles_won")) + _f(pm.get("interceptions")) + _f(pm.get("recoveries"))
                    + _f(pm.get("blocks")) + _f(pm.get("clearances")),
                    _f(r.get("defensive_contribution")),
                )
            )
    out = {}
    for name, values in pairs.items():
        xs = [a for a, _ in values]
        ys = [b for _, b in values]
        out[name] = {"n": len(values), "pearson": _pearson(xs, ys)}
    # team-level: opponent process vs team xGA
    team_rows = _sides(season)
    for name in ("opp_box_shots_allowed", "opp_npxg_allowed", "opp_big_chances_allowed", "opp_shots_allowed"):
        out[f"team {name} vs opponent official xG"] = {
            "n": len(team_rows),
            "pearson": _pearson([r[name] for r in team_rows], [r["xg_against"] for r in team_rows]),
        }
    return out


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "materiality_bars": {
            "team_nll_abs_improvement": TEAM_NLL_ABS_IMPROVEMENT,
            "attacking_rel_improvement": ATTACKING_REL_IMPROVEMENT,
            "fold_fraction": MATERIAL_FOLD_FRACTION,
            "declared": "before any result was inspected",
        }
    }
    rows = _sides("2025-2026")
    print(f"team/opponent: {len(rows)} side-match rows")
    payload["team_opponent"] = evaluate_team(rows)
    print(f"attacking: building 2025-2026 panel")
    payload["attacking"] = evaluate_attacking("2025-2026")
    payload["double_counting"] = double_counting("2025-2026")

    (OUT / "opponent_attacking_evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== TEAM / OPPONENT (Poisson NLL per side-match, lower is better) ===")
    t = payload["team_opponent"]
    print(f"  folds={t['folds']} baseline NLL={t['baseline_nll']:.6f}")
    for name, e in t["candidates"].items():
        print(f"  {name:28s} NLL {e['poisson_nll']:.6f} (d {e['delta_vs_baseline']:+.6f}) "
              f"folds+ {e['folds_improving']}/{e['folds']} -> {e['verdict']}")
    print("\n=== ATTACKING (MAE on realised official xG/xA) ===")
    for name, e in payload["attacking"].items():
        print(f"  {name:34s} base {e['baseline_mae']:.6f} cand {e['candidate_mae']:.6f} "
              f"rel {e['relative_improvement']:+.4%} folds+ {e['folds_improving']}/{e['folds']} -> {e['verdict']}")
    print("\n=== DOUBLE COUNTING (Pearson) ===")
    for name, e in payload["double_counting"].items():
        print(f"  {name:46s} n={e['n']:<6d} r={e['pearson']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
