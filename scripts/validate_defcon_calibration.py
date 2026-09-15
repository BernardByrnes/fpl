"""DEFCON structural calibration — final bounded gate.

Calibrates P(hit) ONLY.  The action-rate model, prior ESS, expected minutes,
threshold definitions and official labels are untouched; this asks whether the
production probability itself can be mapped to something better calibrated.

Candidates: IDENTITY, PLATT, BETA, and (research only) ISOTONIC.

Point-in-time discipline: a row's raw probability is built from STRICTLY earlier
gameweeks (2025/26 within-season; 2026/27 also draws on the full 2025/26 season,
exactly as production draws on prior-season history).  Calibrators are fitted on
prior folds only, and the single "frozen" calibrator per method is fitted on
2025/26 alone.  No 2026/27 label touches any fit, and every frozen prediction is
generated before any current-season metric is inspected.

Pure standard library — the Brain ships without numpy/scipy.
"""

from __future__ import annotations

import csv
import json
import math
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
EPS = 1e-6

PLATT_VERSION = "defcon_platt_v1.0.0"
BETA_VERSION = "defcon_beta_v1.0.0"

POSITION_CODES = {
    "goalkeeper": "GKP", "gkp": "GKP", "defender": "DEF", "def": "DEF",
    "midfielder": "MID", "mid": "MID", "forward": "FWD", "fwd": "FWD",
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


# ---------------------------------------------------------------------------
# Data (unchanged construction from the validation phase)
# ---------------------------------------------------------------------------


def load_2025_26() -> list[dict]:
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
            rows.append({"season": "2025/26", "gw": gw, "player_id": pid, "position": pos,
                         "minutes": minutes, "actions": actions,
                         "hit": int(actions >= int(threshold))})
    return rows


def load_2026_27() -> list[dict]:
    from fpl_brain.database import connect_readonly_database

    conn = connect_readonly_database(LIVE_DB)
    try:
        positions = {
            int(r["id"]): ELEMENT_TYPE_TO_CODE.get(int(r["element_type"] or 0), "")
            for r in conn.execute("SELECT id, element_type FROM players")
        }
        raw = [dict(r) for r in conn.execute(
            """SELECT pg.player_id, pg.event, pg.minutes, pg.defensive_contribution
               FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
               WHERE f.finished = 1 AND f.started = 1 AND pg.event <= 3"""
        )]
    finally:
        conn.close()

    counts: dict[tuple[int, int], int] = defaultdict(int)
    for r in raw:
        if _f(r["minutes"]) > 0:
            counts[(int(r["player_id"]), int(r["event"]))] += 1

    rows: list[dict] = []
    for r in raw:
        pid, gw = int(r["player_id"]), int(r["event"])
        minutes = _f(r["minutes"])
        if minutes <= 0 or counts.get((pid, gw), 0) != 1:
            continue
        pos = positions.get(pid, "")
        threshold = RULES.defcon_threshold_for(pos)
        if threshold is None or pos not in RULES.defcon_positions:
            continue
        if r["defensive_contribution"] is None:
            continue
        actions = float(r["defensive_contribution"])
        rows.append({"season": "2026/27", "gw": gw, "player_id": pid, "position": pos,
                     "minutes": minutes, "actions": actions,
                     "hit": int(actions >= int(threshold))})
    return rows


# ---------------------------------------------------------------------------
# The RAW production signal, built point-in-time
# ---------------------------------------------------------------------------


def raw_probabilities(seed: list[dict], rows: list[dict]) -> list[float]:
    """Production raw P(hit) for each row, using only strictly earlier gameweeks.

    ``seed`` supplies history available before the corpus starts (the prior
    season for a new season).  Nothing from the row's own gameweek or later can
    enter, so this is what production would have produced at that moment.
    """

    own: dict[int, list] = defaultdict(lambda: [0.0, 0.0, ""])
    per_position: dict[str, list[float]] = defaultdict(list)
    for r in seed:
        own[r["player_id"]][0] += r["actions"]
        own[r["player_id"]][1] += r["minutes"]
        own[r["player_id"]][2] = r["position"]
        per_position[r["position"]].append(r["actions"] * 90.0 / max(1.0, r["minutes"]))
    pooled = {pos: sum(v) / len(v) for pos, v in per_position.items() if v}
    stats_rows = list(seed)

    out: list[float] = []
    for gw in sorted({r["gw"] for r in rows}):
        # state is frozen at the start of the gameweek: every row in GW t sees
        # exactly the history from GWs < t, and rows never see each other.
        snap_own = {pid: tuple(v) for pid, v in own.items()}
        snap_pooled = dict(pooled)
        dispersion = {pos: st.dispersion_r for pos, st in dm.fit_count_statistics(stats_rows).items()}
        for r in [x for x in rows if x["gw"] == gw]:
            entry = snap_own.get(r["player_id"], (0.0, 0.0, r["position"]))
            rate = dm.defcon_rate_posterior(
                observed_actions=entry[0], observed_minutes=entry[1],
                prior_rate_per90=snap_pooled.get(entry[2] or r["position"], 0.0),
                prior_ess_minutes=PRIOR_ESS_MINUTES,
            )
            projection = dm.project_defcon(
                position=r["position"], expected_minutes=r["minutes"], rate_posterior=rate,
                variant=dm.VARIANT_BASELINE_POISSON,
                dispersion_r=dispersion.get(r["position"], dm.NB_POISSON_LIMIT_R),
            )
            out.append(projection.p_threshold)
        for r in [x for x in rows if x["gw"] == gw]:
            own[r["player_id"]][0] += r["actions"]
            own[r["player_id"]][1] += r["minutes"]
            own[r["player_id"]][2] = r["position"]
            per_position[r["position"]].append(r["actions"] * 90.0 / max(1.0, r["minutes"]))
            stats_rows.append(r)
        pooled = {pos: sum(v) / len(v) for pos, v in per_position.items() if v}
    return out


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


def _clip(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] if abs(a[i][i]) > 1e-12 else 0.0 for i in range(n)]


def _logistic_fit(design: list[list[float]], outcomes: list[int], *, ridge: float = 1e-8,
                  iterations: int = 100) -> list[float]:
    """Newton-Raphson (IRLS) logistic regression — pure Python.

    Returns weights in DESIGN-COLUMN ORDER.  Callers that pass ``[1.0, x]`` get
    ``[intercept, slope]``; do not reorder them, because a transposed unpack
    yields two plausible-looking numbers that describe the wrong model.
    """

    k = len(design[0])
    w = [0.0] * k
    for _ in range(iterations):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for x, y in zip(design, outcomes):
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = _sigmoid(z)
            wt = max(p * (1.0 - p), 1e-9)
            resid = p - y
            for i in range(k):
                grad[i] += x[i] * resid
                for j in range(k):
                    hess[i][j] += x[i] * x[j] * wt
        for i in range(k):
            hess[i][i] += ridge
        step = _solve(hess, grad)
        move = max(abs(s) for s in step)
        w = [wi - si for wi, si in zip(w, step)]
        if move < 1e-10:
            break
    return w


def platt_fit(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    design = [[1.0, _logit(p)] for p, _ in pairs]
    a, b = _logistic_fit(design, [y for _, y in pairs])
    return a, b


def platt_apply(p: float, params: tuple[float, float]) -> float:
    a, b = params
    return _sigmoid(a + b * _logit(p))


def beta_fit(pairs: list[tuple[float, int]]) -> tuple[float, float, float]:
    design = [[1.0, math.log(_clip(p)), math.log(1.0 - _clip(p))] for p, _ in pairs]
    a, b, c = _logistic_fit(design, [y for _, y in pairs])
    return a, b, c


def beta_apply(p: float, params: tuple[float, float, float]) -> float:
    a, b, c = params
    pc = _clip(p)
    return _sigmoid(a + b * math.log(pc) + c * math.log(1.0 - pc))


def isotonic_fit(pairs: list[tuple[float, int]], *, min_bin: int = 40) -> list[tuple[float, float]]:
    """Pool-adjacent-violators on quantile bins (research comparator only)."""

    ordered = sorted(pairs, key=lambda t: t[0])
    n = len(ordered)
    bins = max(2, min(50, n // min_bin))
    size = max(1, n // bins)
    blocks = []
    for start in range(0, n, size):
        chunk = ordered[start:start + size]
        if not chunk:
            continue
        blocks.append([sum(p for p, _ in chunk) / len(chunk),
                       sum(y for _, y in chunk) / len(chunk), len(chunk)])
    # PAVA
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1):
            if blocks[i][1] > blocks[i + 1][1]:
                total = blocks[i][2] + blocks[i + 1][2]
                merged = [
                    (blocks[i][0] * blocks[i][2] + blocks[i + 1][0] * blocks[i + 1][2]) / total,
                    (blocks[i][1] * blocks[i][2] + blocks[i + 1][1] * blocks[i + 1][2]) / total,
                    total,
                ]
                blocks[i:i + 2] = [merged]
                changed = True
                break
    return [(b[0], b[1]) for b in blocks]


def isotonic_apply(p: float, steps: list[tuple[float, float]]) -> float:
    if not steps:
        return p
    value = steps[0][1]
    for threshold, level in steps:
        if p >= threshold:
            value = level
        else:
            break
    return value


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def _monotone(apply, grid: list[float]) -> bool:
    values = [apply(p) for p in grid]
    return all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def metrics(probs: list[float], hits: list[int]) -> dict:
    brier = dm.brier_score(probs, hits)
    loss = dm.log_loss(probs, hits)
    ep = dm.expected_points_calibration(probs, hits, points=RULES.defcon_points)
    return {"n": len(probs), "brier": brier, "log_loss": loss, **ep,
            "calibration": dm.calibration_table(probs, hits)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {"production_xpts_changed": False,
                     "scope": "calibrates P(hit) only; rate model, ESS, minutes and labels untouched"}

    fit_rows = load_2025_26()
    fit_rows.sort(key=lambda r: r["gw"])
    fit_raw = raw_probabilities([], fit_rows)
    for r, p in zip(fit_rows, fit_raw):
        r["p_raw"] = p
    payload["corpus"] = {
        "2025_26_rows": len(fit_rows),
        "2025_26_gws": sorted({r["gw"] for r in fit_rows}),
        "2025_26_hit_rate": sum(r["hit"] for r in fit_rows) / len(fit_rows),
    }
    grid = [EPS + (1 - 2 * EPS) * i / 200 for i in range(201)]

    # ---- 2025/26 rolling OOS ------------------------------------------------
    rolling: dict[str, dict] = {}
    methods = ("IDENTITY", "PLATT", "BETA", "ISOTONIC")
    pooled_probs: dict[str, list[float]] = {m: [] for m in methods}
    pooled_hits: list[int] = []
    fold_wins: dict[str, int] = {m: 0 for m in methods}
    folds = 0
    for origin in sorted({r["gw"] for r in fit_rows if r["gw"] >= 8}):
        prior = [r for r in fit_rows if r["gw"] < origin]
        test = [r for r in fit_rows if r["gw"] == origin]
        if len(prior) < 500 or not test:
            continue
        folds += 1
        pairs = [(r["p_raw"], r["hit"]) for r in prior]
        platt = platt_fit(pairs)
        beta = beta_fit(pairs)
        iso = isotonic_fit(pairs)
        fold_brier: dict[str, float] = {}
        for name, apply in (
            ("IDENTITY", lambda p: p),
            ("PLATT", lambda p: platt_apply(p, platt)),
            ("BETA", lambda p: beta_apply(p, beta)),
            ("ISOTONIC", lambda p: isotonic_apply(p, iso)),
        ):
            values = [apply(r["p_raw"]) for r in test]
            fold_brier[name] = dm.brier_score(values, [r["hit"] for r in test])
            pooled_probs[name].extend(values)
            if name != "IDENTITY":
                pass
        pooled_hits.extend(r["hit"] for r in test)
        best = min(fold_brier, key=lambda k: fold_brier[k])
        fold_wins[best] += 1

    for name in methods:
        entry = metrics(pooled_probs[name], pooled_hits)
        entry["fold_wins"] = fold_wins[name]
        entry["folds"] = folds
        # Standard logistic calibration: logit(P(y=1)) = intercept + slope*logit(p).
        # ``_logistic_fit`` returns weights in DESIGN-COLUMN order, and the design
        # here is [1.0, x], so w[0] is the INTERCEPT (the constant column) and
        # w[1] is the SLOPE.  Unpacking these the other way round silently
        # transposes the two statistics — which is exactly what happened on the
        # first pass and produced an apparent "negative calibration slope".
        design = [[1.0, _logit(p)] for p in pooled_probs[name]]
        intercept, slope = _logistic_fit(design, pooled_hits)
        entry["calibration_slope"] = slope
        entry["calibration_intercept"] = intercept
        rolling[name] = entry
    payload["rolling_2025_26"] = rolling

    print("=== 2025/26 ROLLING OOS (fit on prior folds only) ===")
    print("  %-10s %-10s %-10s %-12s %-9s %s" % ("method", "Brier", "log loss", "ep bias", "folds won", "slope/intercept"))
    for name in methods:
        e = rolling[name]
        print("  %-10s %.6f  %.6f  %+.6f   %-9s %.3f / %+.3f" % (
            name, e["brier"], e["log_loss"], e["signed_bias_points"],
            f"{e['fold_wins']}/{e['folds']}", e["calibration_slope"], e["calibration_intercept"]))

    # ---- frozen calibrators from 2025/26 ONLY -------------------------------
    all_pairs = [(r["p_raw"], r["hit"]) for r in fit_rows]
    frozen_platt = platt_fit(all_pairs)
    frozen_beta = beta_fit(all_pairs)
    frozen_iso = isotonic_fit(all_pairs)
    payload["frozen_2025_26_calibrators"] = {
        "PLATT": {"version": PLATT_VERSION, "a": frozen_platt[0], "b": frozen_platt[1],
                  "form": "sigmoid(a + b * logit(p_raw))",
                  "monotonic": _monotone(lambda p: platt_apply(p, frozen_platt), grid)},
        "BETA": {"version": BETA_VERSION, "a": frozen_beta[0], "b": frozen_beta[1], "c": frozen_beta[2],
                 "form": "sigmoid(a + b*log(p_raw) + c*log(1-p_raw))",
                 "monotonic": _monotone(lambda p: beta_apply(p, frozen_beta), grid)},
        "ISOTONIC_steps": len(frozen_iso),
        "fitted_on_rows": len(fit_rows),
    }
    print("\n=== FROZEN 2025/26 CALIBRATORS ===")
    print("  PLATT  a=%.6f b=%.6f  monotonic=%s  (b>0 is required for monotone increasing)"
          % (frozen_platt[0], frozen_platt[1],
             payload["frozen_2025_26_calibrators"]["PLATT"]["monotonic"]))
    print("  BETA   a=%.6f b=%.6f c=%.6f  monotonic=%s"
          % (frozen_beta[0], frozen_beta[1], frozen_beta[2],
             payload["frozen_2025_26_calibrators"]["BETA"]["monotonic"]))
    print("  ISOTONIC %d steps (research only)" % len(frozen_iso))

    # ---- 2026/27 untouched holdout -----------------------------------------
    hold = load_2026_27()
    hold.sort(key=lambda r: r["gw"])
    hold_raw = raw_probabilities(fit_rows, hold)
    for r, p in zip(hold, hold_raw):
        r["p_raw"] = p
    hits = [r["hit"] for r in hold]
    applied = {
        "IDENTITY": [r["p_raw"] for r in hold],
        "PLATT": [platt_apply(r["p_raw"], frozen_platt) for r in hold],
        "BETA": [beta_apply(r["p_raw"], frozen_beta) for r in hold],
        "ISOTONIC": [isotonic_apply(r["p_raw"], frozen_iso) for r in hold],
    }
    holdout: dict[str, dict] = {}
    identity = applied["IDENTITY"]
    for name, values in applied.items():
        entry = metrics(values, hits)
        entry["mean_predicted_probability"] = sum(values) / len(values)
        entry["rows_better"] = sum(1 for a, b in zip(identity, values)
                                   if (a - y) ** 2 > (b - y) ** 2 for y in [0]) if False else None
        better = sum(1 for a, b, y in zip(identity, values, hits) if (b - y) ** 2 < (a - y) ** 2)
        worse = sum(1 for a, b, y in zip(identity, values, hits) if (b - y) ** 2 > (a - y) ** 2)
        entry["rows_better"] = better
        entry["rows_worse"] = worse
        holdout[name] = entry
    ranks = {
        "PLATT": _spearman([r["p_raw"] for r in hold], applied["PLATT"]),
        "BETA": _spearman([r["p_raw"] for r in hold], applied["BETA"]),
        "ISOTONIC": _spearman([r["p_raw"] for r in hold], applied["ISOTONIC"]),
    }
    payload["holdout_2026_27"] = {
        "rows": len(hold),
        "rows_by_gw": {str(g): sum(1 for r in hold if r["gw"] == g) for g in sorted({r["gw"] for r in hold})},
        "observed_hit_rate": sum(hits) / len(hits),
        "results": holdout,
        "rank_correlation_vs_raw": ranks,
    }

    print("\n=== 2026/27 UNTOUCHED HOLDOUT (frozen 2025/26 calibrators, no refit) ===")
    print("  rows %d  observed %.4f" % (len(hold), sum(hits) / len(hits)))
    print("  %-10s %-10s %-10s %-12s %-12s %s" % ("method", "Brier", "log loss", "mean pred", "ep bias", "better/worse"))
    for name in methods:
        e = holdout[name]
        print("  %-10s %.6f  %.6f  %.4f       %+.6f   %d/%d" % (
            name, e["brier"], e["log_loss"], e["mean_predicted_probability"],
            e["signed_bias_points"], e["rows_better"], e["rows_worse"]))
    print("\n  rank correlation raw vs calibrated:",
          {k: round(v, 4) for k, v in ranks.items()})

    # ---- promotion bar ------------------------------------------------------
    base_roll, base_hold = rolling["IDENTITY"], holdout["IDENTITY"]
    verdict = {}
    for name in ("PLATT", "BETA", "ISOTONIC"):
        r, h = rolling[name], holdout[name]
        checks = {
            "improves_both_metrics_2025_26": r["brier"] < base_roll["brier"] and r["log_loss"] < base_roll["log_loss"],
            "improves_both_metrics_2026_27": h["brier"] < base_hold["brier"] and h["log_loss"] < base_hold["log_loss"],
            "reduces_expected_points_bias": abs(h["signed_bias_points"]) < abs(base_hold["signed_bias_points"]),
            "no_new_tail_failure": max(
                abs(row["mean_predicted"] - row["observed_rate"])
                for row in h["calibration"] if row["n"] >= 30
            ) <= max(
                abs(row["mean_predicted"] - row["observed_rate"])
                for row in base_hold["calibration"] if row["n"] >= 30
            ),
            "monotonic": (_monotone(lambda p: platt_apply(p, frozen_platt), grid) if name == "PLATT"
                          else _monotone(lambda p: beta_apply(p, frozen_beta), grid) if name == "BETA"
                          else _monotone(lambda p: isotonic_apply(p, frozen_iso), grid)),
        }
        checks["PASS"] = all(checks.values())
        verdict[name] = checks
    payload["promotion_bar"] = verdict
    print("\n=== PROMOTION BAR ===")
    for name, checks in verdict.items():
        flag = "PASS" if checks["PASS"] else "FAIL"
        detail = ",".join(k for k, v in checks.items() if k != "PASS" and not v) or "all criteria met"
        print("  %-10s %s  (%s)" % (name, flag, detail))

    (OUT / "defcon_structural_calibration.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
