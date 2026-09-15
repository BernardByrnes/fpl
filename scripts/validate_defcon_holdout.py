"""DEFCON final validation: lock the 2025/26 calibration, then hold out 2026/27.

TWO STRICTLY SEPARATED STAGES.

Stage 1 (fit) uses ONLY 2025/26.  Every parameter the candidate would ship —
the level scale and the per-position dispersion — is derived there and then
FROZEN.  No 2026/27 observation influences fitting, tuning, model selection,
threshold selection or parameter choice.

Stage 2 (evaluate) applies those frozen parameters, untouched, to 2026/27
GW1-GW3 realised official outcomes.  The labels come from the Brain's OWN
database (first-party official), not from the external carrier, so the holdout
label provenance is not in question.

Nothing here writes to the live database and no production formula is changed.
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
PRIOR_ESS_MINUTES = 600.0

POSITION_CODES = {
    "goalkeeper": "GKP", "gkp": "GKP",
    "defender": "DEF", "def": "DEF",
    "midfielder": "MID", "mid": "MID",
    "forward": "FWD", "fwd": "FWD",
}
ELEMENT_TYPE_TO_CODE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


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


def load_2025_26() -> list[dict]:
    """FIT-ONLY corpus.  Player-match rows, Premier League, single-match GWs."""

    base = EXTERNAL_ROOT / "2025-2026" / "By Tournament" / "Premier League"
    rows: list[dict] = []
    for gw_dir in sorted(base.glob("GW*"), key=lambda p: int(p.name[2:])):
        gw = int(gw_dir.name[2:])
        stats = _read(gw_dir / "player_gameweek_stats.csv")
        player_matches = _read(gw_dir / "playermatchstats.csv")
        roster = _read(gw_dir / "players.csv")
        if not (stats and player_matches and roster):
            continue
        position = {
            int(_f(r["player_id"])): POSITION_CODES.get(str(r.get("position") or "").lower(), "")
            for r in roster if _f(r["player_id"])
        }
        per_player: dict[int, int] = defaultdict(int)
        for r in player_matches:
            pid = int(_f(r["player_id"]))
            if pid and _f(r.get("minutes_played")) > 0:
                per_player[pid] += 1
        for r in stats:
            pid = int(_f(r.get("player_id") or r.get("id")))
            minutes = _f(r.get("minutes"))
            if not pid or minutes <= 0 or per_player.get(pid, 0) != 1:
                continue
            pos = position.get(pid, "")
            threshold = RULES.defcon_threshold_for(pos)
            if threshold is None or pos not in RULES.defcon_positions:
                continue
            actions = _f(r.get("defensive_contribution"))
            rows.append({
                "season": "2025/26", "gw": gw, "player_id": pid, "position": pos,
                "minutes": minutes, "actions": actions, "hit": int(actions >= int(threshold)),
                "threshold": int(threshold),
            })
    return rows


def load_2026_27_holdout() -> list[dict]:
    """HOLDOUT corpus straight from the Brain's own official database."""

    from fpl_brain.database import connect_readonly_database

    conn = connect_readonly_database(LIVE_DB)
    try:
        positions = {
            int(r["id"]): ELEMENT_TYPE_TO_CODE.get(int(r["element_type"] or 0), "")
            for r in conn.execute("SELECT id, element_type FROM players")
        }
        rows: list[dict] = []
        for r in conn.execute(
            """SELECT pg.player_id, pg.event, pg.fixture_id, pg.minutes,
                      pg.defensive_contribution
               FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
               WHERE f.finished = 1 AND f.started = 1 AND pg.event <= 3"""
        ):
            rows.append(dict(r))
    finally:
        conn.close()

    per_player_gw: dict[tuple[int, int], int] = defaultdict(int)
    for r in rows:
        if _f(r["minutes"]) > 0:
            per_player_gw[(int(r["player_id"]), int(r["event"]))] += 1

    out: list[dict] = []
    for r in rows:
        pid, event = int(r["player_id"]), int(r["event"])
        minutes = _f(r["minutes"])
        if minutes <= 0 or per_player_gw.get((pid, event), 0) != 1:
            continue
        pos = positions.get(pid, "")
        threshold = RULES.defcon_threshold_for(pos)
        if threshold is None or pos not in RULES.defcon_positions:
            continue
        actions = r["defensive_contribution"]
        if actions is None:
            continue
        out.append({
            "season": "2026/27", "gw": event, "player_id": pid, "position": pos,
            "minutes": minutes, "actions": float(actions),
            "hit": int(float(actions) >= int(threshold)), "threshold": int(threshold),
        })
    return out


# ---------------------------------------------------------------------------
# Stage 1 — fit and FREEZE (2025/26 only)
# ---------------------------------------------------------------------------


def pooled_prior_rates(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grouped[r["position"]].append(r["actions"] * 90.0 / max(1.0, r["minutes"]))
    return {pos: sum(v) / len(v) for pos, v in grouped.items() if v}


def own_history(rows: list[dict]) -> dict[int, tuple[float, float, str]]:
    totals: dict[int, list] = defaultdict(lambda: [0.0, 0.0, ""])
    for r in rows:
        totals[r["player_id"]][0] += r["actions"]
        totals[r["player_id"]][1] += r["minutes"]
        totals[r["player_id"]][2] = r["position"]
    return {pid: (v[0], v[1], v[2]) for pid, v in totals.items()}


def rates_for(rows: list[dict], prior: dict[int, tuple[float, float, str]],
              pooled: dict[str, float]) -> list[float]:
    out = []
    for r in rows:
        acts, mins, pos = prior.get(r["player_id"], (0.0, 0.0, r["position"]))
        out.append(dm.defcon_rate_posterior(
            observed_actions=acts, observed_minutes=mins,
            prior_rate_per90=pooled.get(pos, 0.0), prior_ess_minutes=PRIOR_ESS_MINUTES,
        ))
    return out


def fit_scale(rows: list[dict], rates: list[float], dispersion: dict[str, float],
              variant: str, *, objective: str = "brier") -> float:
    prepared = []
    for r, rate in zip(rows, rates):
        projection = dm.project_defcon(
            position=r["position"], expected_minutes=r["minutes"], rate_posterior=rate,
            variant=variant, dispersion_r=dispersion.get(r["position"], dm.NB_POISSON_LIMIT_R),
            level_scale=1.0,
        )
        if projection.threshold is None:
            continue
        prepared.append((projection.projected_defcon, projection.threshold, r["hit"],
                         projection.dispersion_r))
    if not prepared:
        return 1.0
    observed = sum(h for _, _, h, _ in prepared) / len(prepared)
    poisson_like = variant == dm.VARIANT_LEVEL_CALIBRATED_POISSON

    def objective_value(scale: float) -> float:
        total = 0.0
        for mean, threshold, hit, r in prepared:
            p = (dm.xpts.poisson_tail_probability(mean * scale, threshold) if poisson_like
                 else dm.negative_binomial_tail(mean * scale, r, threshold))
            total += (p - hit) ** 2 if objective == "brier" else p
        return total / len(prepared) if objective == "brier" else abs(total / len(prepared) - observed)

    lo, hi = 0.05, 4.0
    best = min((lo + (hi - lo) * i / 40 for i in range(41)), key=objective_value)
    lo, hi = max(0.02, best * 0.9), best * 1.1
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if objective_value(m1) < objective_value(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Stage 2 — evaluate (frozen parameters, 2026/27 holdout)
# ---------------------------------------------------------------------------


def score(rows: list[dict], probabilities: dict[str, list[float]]) -> dict:
    out = {}
    for name, probs in probabilities.items():
        outcomes = [r["hit"] for r in rows]
        out[name] = {
            "n": len(probs),
            "brier": dm.brier_score(probs, outcomes),
            "log_loss": dm.log_loss(probs, outcomes),
            **dm.expected_points_calibration(probs, outcomes, points=RULES.defcon_points),
            "calibration": dm.calibration_table(probs, outcomes),
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "production_xpts_changed": False,
        "claim_language": (
            "Prior-season component-level calibration bias in the current DEFCON "
            "formula, with strong rolling-OOS evidence. End-to-end current-season "
            "production improvement remains unproven."
        ),
    }

    # ---- STAGE 1: fit on 2025/26 ONLY ------------------------------------
    fit_rows = load_2025_26()
    fit_pooled = pooled_prior_rates(fit_rows)
    fit_own = own_history(fit_rows)
    fit_rates = rates_for(fit_rows, fit_own, fit_pooled)
    fit_stats = dm.fit_count_statistics(fit_rows)
    dispersion = {pos: st.dispersion_r for pos, st in fit_stats.items()}

    scale_poisson = fit_scale(fit_rows, fit_rates, dispersion, dm.VARIANT_LEVEL_CALIBRATED_POISSON)
    scale_nb = fit_scale(fit_rows, fit_rates, dispersion, dm.VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL)
    scale_poisson_mean = fit_scale(fit_rows, fit_rates, dispersion,
                                   dm.VARIANT_LEVEL_CALIBRATED_POISSON, objective="mean")
    scale_nb_mean = fit_scale(fit_rows, fit_rates, dispersion,
                              dm.VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL, objective="mean")

    payload["stage1_locked_parameters"] = {
        "fitted_on": "2025/26 only",
        "rows": len(fit_rows),
        "poisson_level_scale_brier_fitted": scale_poisson,
        "negative_binomial_level_scale_brier_fitted": scale_nb,
        "poisson_level_scale_mean_matched": scale_poisson_mean,
        "negative_binomial_level_scale_mean_matched": scale_nb_mean,
        "dispersion_r_by_position": dispersion,
        "dispersion_note": f"r = {dm.NB_POISSON_LIMIT_R:g} means the Poisson limit",
        "prior_ess_minutes": PRIOR_ESS_MINUTES,
    }
    print("=== STAGE 1: 2025/26 LOCKED PARAMETERS ===")
    print(f"  fit rows                        : {len(fit_rows)}")
    print(f"  POISSON level scale (Brier)     : {scale_poisson:.6f}")
    print(f"  NEG-BIN level scale (Brier)     : {scale_nb:.6f}")
    print(f"  POISSON level scale (mean match): {scale_poisson_mean:.6f}")
    print(f"  NEG-BIN level scale (mean match): {scale_nb_mean:.6f}")
    print(f"  dispersion r by position        : { {k: round(v,3) for k,v in dispersion.items()} }")

    # ---- STAGE 2: 2026/27 holdout, frozen parameters ---------------------
    holdout = load_2026_27_holdout()
    by_gw: dict[int, int] = defaultdict(int)
    for r in holdout:
        by_gw[r["gw"]] += 1
    print(f"\n=== STAGE 2: 2026/27 HOLDOUT (official Brain DB) ===")
    print(f"  rows {len(holdout)} across GW{sorted(by_gw)} -> {dict(sorted(by_gw.items()))}")

    # prior evidence = full 2025/26 + the current season's own earlier GWs
    probabilities: dict[str, list[float]] = defaultdict(list)
    for gw in sorted(by_gw):
        test = [r for r in holdout if r["gw"] == gw]
        prior_rows = fit_rows + [r for r in holdout if r["gw"] < gw]
        pooled = pooled_prior_rates(prior_rows)
        own = own_history(prior_rows)
        rates = rates_for(test, own, pooled)
        for r, rate in zip(test, rates):
            r["_rate"] = rate
            disp = dispersion.get(r["position"], dm.NB_POISSON_LIMIT_R)
            for name, variant, scale in (
                ("CURRENT_FORMULA", dm.VARIANT_BASELINE_POISSON, 1.0),
                ("CALIBRATED_POISSON", dm.VARIANT_LEVEL_CALIBRATED_POISSON, scale_poisson),
                ("CALIBRATED_NEGATIVE_BINOMIAL", dm.VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL, scale_nb),
            ):
                projection = dm.project_defcon(
                    position=r["position"], expected_minutes=r["minutes"],
                    rate_posterior=rate, variant=variant, dispersion_r=disp,
                    level_scale=scale, player_id=r["player_id"],
                )
                probabilities[name].append(projection.p_threshold)

    results = score(holdout, probabilities)

    # Paired per-row comparison: each row is scored by every model on the SAME
    # observation, so the difference in squared error is paired and its standard
    # error is meaningful even though the sample is only three gameweeks.
    outcomes = [r["hit"] for r in holdout]
    paired = {}
    for name, probs in probabilities.items():
        if name == "CURRENT_FORMULA":
            continue
        diffs = [(p0 - y) ** 2 - (p1 - y) ** 2
                 for p0, p1, y in zip(probabilities["CURRENT_FORMULA"], probs, outcomes)]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        se = math.sqrt(var / len(diffs)) if var > 0 else 0.0
        paired[name] = {
            "mean_brier_improvement_per_row": mean,
            "standard_error": se,
            "t_statistic": (mean / se) if se > 0 else None,
            "rows_improved": sum(1 for d in diffs if d > 0),
            "rows_worsened": sum(1 for d in diffs if d < 0),
            "n": len(diffs),
        }
    payload["stage2_paired"] = paired
    payload["stage2_holdout"] = {
        "rows": len(holdout),
        "rows_by_gw": dict(sorted(by_gw.items())),
        "refit_on_holdout": False,
        "label_source": "Brain official database (player_gameweeks.defensive_contribution, fixtures finished+started)",
        "results": results,
    }

    base = results["CURRENT_FORMULA"]
    print("\n  model                            Brier     log loss  mean pred  observed  ep bias")
    for name, entry in results.items():
        print("  %-32s %.6f  %.6f  %.4f     %.4f    %+.4f%s" % (
            name, entry["brier"], entry["log_loss"], entry["mean_predicted_probability"],
            entry["realised_hit_rate"], entry["signed_bias_points"],
            "   <- baseline" if name == "CURRENT_FORMULA" else ""))
    print()
    for name, entry in results.items():
        if name == "CURRENT_FORMULA":
            continue
        print("  %-32s d Brier %+.6f (%+.2f%%)  d logloss %+.6f  d bias %+.4f" % (
            name, entry["brier"] - base["brier"],
            100 * (entry["brier"] - base["brier"]) / base["brier"],
            entry["log_loss"] - base["log_loss"],
            entry["signed_bias_points"] - base["signed_bias_points"]))

    print("\n  paired per-row Brier improvement vs CURRENT_FORMULA:")
    for name, entry in paired.items():
        t = entry["t_statistic"]
        print("  %-32s mean %+.6f  SE %.6f  t=%s  better/worse %d/%d" % (
            name, entry["mean_brier_improvement_per_row"], entry["standard_error"],
            "n/a" if t is None else "%.2f" % t, entry["rows_improved"], entry["rows_worsened"]))

    (OUT / "defcon_final_validation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
