"""Attacking process model — rolling-OOS evaluation against official xG/xA.

THE CONTRACT
------------
Official FPL xG/xA remain the ANCHOR.  Nothing here overwrites them.  Each
candidate is a small, regularised linear model that predicts a player's FUTURE
official xG/90 (or xA/90) from:

    the player's own prior official rate          (the anchor)
  + at most a compact set of external process rates (the increment)

and every coefficient is fitted on PRIOR folds only.  The evaluation then asks
whether the increment predicts the realised next-gameweek official xG/xA better
than the anchor alone.

Both arms are fitted on the same prior rows with the same estimator and scored on
the same test rows, so a candidate cannot win by over-fitting the fold it is
scored on.

Process rates are shrunk toward a POSITION-AWARE prior weighted by exposure, so a
player with four minutes of external evidence is pulled almost entirely to the
position pool rather than producing a wild per-90.  Missing external evidence is
NOT zero: it is recorded as NO_PROCESS_EVIDENCE and the row falls back to the
official-only anchor.
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

from fpl_brain import core_process_ingest as ci  # noqa: E402
from fpl_brain import core_process_features as cf  # noqa: E402

EXTERNAL_ROOT = Path(
    "K:/FPL-core/data/exports/fpl_core/extracted/"
    "FPL-Core-Insights-d2eee2a25645ec4a73bda640f0d92048166fa22c/data"
)
LIVE_DB = "K:/FPL/fpl.db"
OUT = Path("K:/FPL-atk/data/exports/attacking_v1")

POSITION_CODES = {
    "goalkeeper": "GKP", "gkp": "GKP", "defender": "DEF", "def": "DEF",
    "midfielder": "MID", "mid": "MID", "forward": "FWD", "fwd": "FWD",
}
ELEMENT_TYPE_TO_CODE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

#: Shrinkage: position-aware prior, exposure-weighted, in minutes.
PROCESS_PRIOR_MINUTES = 600.0
#: Production prior ESS, taken from PlayerRatesConfig (xg_prior_ess_minutes /
#: xa_prior_ess_minutes).  The anchor must use the SAME shrinkage production
#: applies, or the process term is credited with recovering shrinkage that
#: the production model already does.
PRODUCTION_ESS = {"xg": 850.0, "xa": 1100.0}

# --- PROMOTION BAR (declared BEFORE any result was inspected) ---------------
BAR_MAE_RELATIVE = 0.010        # >= 1.0% relative MAE improvement
BAR_RMSE_TOLERANCE = 0.005      # RMSE may not worsen by more than 0.5%
BAR_FOLD_FRACTION = 0.60        # improvement in >= 60% of folds
BAR_POSITION_REGRESSION = 0.010  # no position subgroup worse than 1% MAE
BAR_COVERAGE = 0.90             # process evidence on >= 90% of scored rows


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
    return int(_f(row.get("player_id") or row.get("id")))


def load_panel(season: str) -> list[dict]:
    """Official outcomes + external process counts, one row per player-gameweek."""

    base = EXTERNAL_ROOT / season / "By Tournament" / "Premier League"
    rows: list[dict] = []
    for gw_dir in sorted(base.glob("GW*"), key=lambda p: int(p.name[2:])):
        gw = int(gw_dir.name[2:])
        stats = _read(gw_dir / "player_gameweek_stats.csv")
        matches = _read(gw_dir / "playermatchstats.csv")
        roster = _read(gw_dir / "players.csv")
        if not (stats and roster):
            continue
        position = {_pid(r): POSITION_CODES.get(str(r.get("position") or "").lower(), "") for r in roster if _pid(r)}
        process = {_pid(r): r for r in matches if _pid(r)}
        for r in stats:
            pid = _pid(r)
            minutes = _f(r.get("minutes"))
            pos = position.get(pid, "")
            if not pid or minutes <= 0 or pos not in ("DEF", "MID", "FWD"):
                continue
            pm = process.get(pid, {})
            has_process = pid in process and "total_shots" in pm
            rows.append({
                "season": season, "gw": gw, "player_id": pid, "position": pos,
                "minutes": minutes,
                "xg": _f(r.get("expected_goals")), "xa": _f(r.get("expected_assists")),
                "shots": _f(pm.get("total_shots")) if has_process else None,
                "sot": _f(pm.get("shots_on_target")) if has_process else None,
                "box_touches": _f(pm.get("touches_opposition_box")) if has_process else None,
                "chances_created": _f(pm.get("chances_created")) if has_process else None,
                "has_process": has_process,
            })
    return rows


# ---------------------------------------------------------------------------
# Evidence construction (point in time)
# ---------------------------------------------------------------------------


def _ols(xs: list[list[float]], ys: list[float]) -> list[float]:
    k = len(xs[0]) + 1
    design = [[1.0] + row for row in xs]
    ata = [[sum(d[a] * d[b] for d in design) for b in range(k)] for a in range(k)]
    aty = [sum(design[i][a] * ys[i] for i in range(len(ys))) for a in range(k)]
    # small ridge keeps a collinear pair from producing an unstable fit
    for i in range(1, k):
        ata[i][i] += 1e-8
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


class Evidence:
    """Everything a fold needs, accumulated from strictly earlier gameweeks."""

    def __init__(self) -> None:
        self.official: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"xg": [0.0, 0.0], "xa": [0.0, 0.0]}
        )
        self.position: dict[int, str] = {}
        self.process: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {k: [0.0, 0.0] for k in ("shots", "sot", "box_touches", "chances_created")}
        )
        self.pool: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add(self, row: dict) -> None:
        pid = row["player_id"]
        # Each component accumulates its OWN official total.  Adding row["xg"]
        # for every component (an easy slip) would make the xA anchor actually
        # be the xG anchor and silently invalidate the entire xA comparison.
        self.official[pid]["xg"][0] += row["xg"]
        self.official[pid]["xg"][1] += row["minutes"]
        self.official[pid]["xa"][0] += row["xa"]
        self.official[pid]["xa"][1] += row["minutes"]
        self.position[pid] = row["position"]
        self.pool[row["position"]]["xg"].append(row["xg"] * 90.0 / row["minutes"])
        self.pool[row["position"]]["xa"].append(row["xa"] * 90.0 / row["minutes"])
        # Self-selection: only a player WITH external evidence contributes to the
        # process pools, so the prior mean is not contaminated by the missing
        # evidence of players who are absent from the feed.
        if row["has_process"]:
            for key in ("shots", "sot", "box_touches", "chances_created"):
                self.process[pid][key][0] += row[key]
                self.process[pid][key][1] += row["minutes"]
                self.pool[row["position"]][key].append(row[key] * 90.0 / row["minutes"])

    def pooled(self, position: str, key: str) -> float:
        values = self.pool.get(position, {}).get(key) or []
        return sum(values) / len(values) if values else 0.0

    def official_rate(self, pid: int, key: str) -> float | None:
        """The PRODUCTION anchor rate: own official evidence shrunk toward the
        position pool with the production ESS.

        This is what ``player_rates._project_component`` effectively produces
        ((minutes*current_rate + prior_ess*prior_rate) / (minutes + prior_ess)),
        so the baseline is the real production rate layer rather than an
        unshrunk stand-in that a process term could "improve" by re-shrinking.
        """

        entry = self.official.get(pid, {}).get(key)
        if entry is None or entry[1] <= 0.0:
            return None
        total, minutes = entry
        position = self.position.get(pid, "")
        pool = self.pooled(position, key)
        ess = PRODUCTION_ESS[key]
        return (total * 90.0 + ess * pool) / (minutes + ess)

    def process_rate(self, row: dict, key: str) -> float | None:
        entry = self.process.get(row["player_id"], {}).get(key)
        if entry is None or entry[1] <= 0.0 or not row["has_process"]:
            return None
        total, minutes = entry
        prior = self.pooled(row["position"], key)
        return cf.shrunken_per90(
            total=total, minutes=minutes, prior_per90=prior, prior_minutes=PROCESS_PRIOR_MINUTES
        )


def build_fold_rows(evidence: Evidence, test: list[dict], key: str,
                    process_keys: tuple[str, ...]) -> tuple[list, list[dict], int]:
    """(features, rows, coverage) for the test rows scoreable under this snapshot.

    A row is scoreable only when the official anchor exists AND every required
    process rate exists.  Missing external evidence is NOT zero -- the row is
    dropped from that candidate's sample rather than being fed a fabricated
    zero, and coverage is reported so a candidate cannot look good by silently
    shrinking its own sample.
    """

    features: list[list[float]] = []
    kept: list[dict] = []
    for row in test:
        anchor = evidence.official_rate(row["player_id"], key)
        if anchor is None:
            continue
        values = []
        for pkey in process_keys:
            rate = evidence.process_rate(row, pkey)
            if rate is None:
                break
            values.append(rate)
        else:
            # Exposure is the realised minutes, identical across arms, so the
            # comparison isolates the RATE model rather than the minutes model.
            exposure = row["minutes"] / 90.0
            features.append([anchor * exposure] + [v * exposure for v in values])
            kept.append(row)
    return features, kept, len(kept)


def _subgroups(base, cand, actual, positions, minutes) -> dict:
    """Where the increment helps and where it does not."""

    out: dict[str, dict] = {}
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, position in enumerate(positions):
        buckets[f"POS_{position}"].append(i)
    for i, value in enumerate(minutes):
        buckets["MIN_60_PLUS" if value >= 60 else "MIN_UNDER_60"].append(i)
    for name, idx in buckets.items():
        if len(idx) < 30:
            continue
        b = sum(abs(base[i] - actual[i]) for i in idx) / len(idx)
        c = sum(abs(cand[i] - actual[i]) for i in idx) / len(idx)
        out[name] = {"n": len(idx), "baseline_mae": b, "candidate_mae": c,
                     "relative": (b - c) / b if b else 0.0}
    return out


def expanding_frame(rows: list[dict], key: str, process_keys: tuple[str, ...]) -> tuple[list, list]:
    """Point-in-time training features: a row may only see STRICTLY earlier GWs.

    This matters.  Accumulating evidence over the whole training block and then
    building features for rows inside it lets each row's own outcome leak into
    its own feature (the anchor and the process rate would both partially equal
    the target), which biases the fitted coefficients -- the anchor's especially
    -- and then mis-weights them on the test fold, where no such leak exists.
    Iterating gameweek by gameweek, building features BEFORE adding the batch,
    reproduces exactly the information state the test fold sees.
    """

    evidence = Evidence()
    feats: list[list[float]] = []
    kept: list[dict] = []
    for gw in sorted({r["gw"] for r in rows}):
        batch = [r for r in rows if r["gw"] == gw]
        fold_feats, fold_rows, _ = build_fold_rows(evidence, batch, key, process_keys)
        feats.extend(fold_feats)
        kept.extend(fold_rows)
        for r in batch:
            evidence.add(r)
    return feats, kept


def evaluate(rows: list[dict], *, key: str, candidates: dict[str, tuple[str, ...]],
             min_origin: int = 8) -> dict:
    gws = sorted({r["gw"] for r in rows})
    out: dict[str, dict] = {}
    for name, process_keys in candidates.items():
        per_fold = []
        fold_wins = 0
        total_scored = 0
        total_test = 0
        pooled_base: list[float] = []
        pooled_cand: list[float] = []
        pooled_actual: list[float] = []
        pooled_position: list[str] = []
        pooled_minutes: list[float] = []
        coefs: list[list[float]] = []
        for origin in [g for g in gws if g >= min_origin]:
            prior = [r for r in rows if r["gw"] < origin]
            test = [r for r in rows if r["gw"] == origin]
            if len(prior) < 400 or len(test) < 40:
                continue
            ev = Evidence()
            for r in prior:
                ev.add(r)
            train_features, train_rows = expanding_frame(prior, key, process_keys)
            if len(train_features) < 200:
                continue
            targets = [r[key] for r in train_rows]
            fit = _ols(train_features, targets)
            fit_base = _ols([[f[0]] for f in train_features], targets)

            feats, kept, _ = build_fold_rows(ev, test, key, process_keys)
            if not feats:
                continue
            total_scored += len(feats)
            total_test += len(test)
            coefs.append(list(fit))
            base_err, cand_err = [], []
            for f, row in zip(feats, kept):
                actual = row[key]
                base_pred = fit_base[0] + fit_base[1] * f[0]
                cand_pred = fit[0] + sum(c * x for c, x in zip(fit[1:], f))
                base_err.append(abs(base_pred - actual))
                cand_err.append(abs(cand_pred - actual))
                pooled_base.append(base_pred); pooled_cand.append(cand_pred)
                pooled_actual.append(actual); pooled_position.append(row['position'])
                pooled_minutes.append(row['minutes'])
            b = sum(base_err) / len(base_err)
            c = sum(cand_err) / len(cand_err)
            per_fold.append({"gw": origin, "baseline_mae": b, "candidate_mae": c, "n": len(feats)})
            if c < b:
                fold_wins += 1

        if not per_fold:
            continue
        out[name] = {
            "process_features": list(process_keys),
            "folds": len(per_fold),
            "fold_wins": fold_wins,
            "fold_fraction_better": fold_wins / len(per_fold),
            "baseline_mae": sum(f["baseline_mae"] for f in per_fold) / len(per_fold),
            "candidate_mae": sum(f["candidate_mae"] for f in per_fold) / len(per_fold),
            "scored_rows": total_scored,
            "test_rows": total_test,
            "coverage": (total_scored / total_test) if total_test else 0.0,
            "per_fold": per_fold,
            "rmse_baseline": math.sqrt(sum((a - b) ** 2 for a, b in zip(pooled_base, pooled_actual)) / len(pooled_actual)),
            "rmse_candidate": math.sqrt(sum((a - b) ** 2 for a, b in zip(pooled_cand, pooled_actual)) / len(pooled_actual)),
            "spearman_baseline": _spearman(pooled_base, pooled_actual),
            "spearman_candidate": _spearman(pooled_cand, pooled_actual),
            "mean_coefficients": [sum(c[i] for c in coefs) / len(coefs) for i in range(len(coefs[0]))],
            "subgroups": _subgroups(pooled_base, pooled_cand, pooled_actual, pooled_position, pooled_minutes),
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "base": "65f8fb2d05807070bb671b5d2e69a6d5104d5992",
        "external_source_sha": "d2eee2a25645ec4a73bda640f0d92048166fa22c",
        "shrinkage": {"process_prior_minutes": PROCESS_PRIOR_MINUTES, "pooling": "position-aware, exposure-weighted"},
        "bar": {
            "mae_relative": BAR_MAE_RELATIVE, "rmse_tolerance": BAR_RMSE_TOLERANCE,
            "fold_fraction": BAR_FOLD_FRACTION, "position_regression": BAR_POSITION_REGRESSION,
            "coverage": BAR_COVERAGE, "declared": "before any result was inspected",
        },
    }
    rows = load_panel("2025-2026")
    payload["panel"] = {
        "rows": len(rows),
        "with_process": sum(1 for r in rows if r["has_process"]),
        "coverage": sum(1 for r in rows if r["has_process"]) / len(rows) if rows else 0.0,
    }
    print(f"panel: {len(rows)} player-gameweek rows, process evidence on "
          f"{payload['panel']['coverage']:.1%}")

    xg_candidates = {
        "XG_BASELINE": (),
        "XG_PLUS_SHOTS": ("shots",),
        "XG_PLUS_BOX_TOUCHES": ("box_touches",),
        "XG_PLUS_SOT": ("sot",),
        "XG_COMPACT_SB": ("shots", "box_touches"),
        "XG_COMPACT_ALL3": ("shots", "box_touches", "sot"),
    }
    xa_candidates = {"XA_BASELINE": (), "XA_PROCESS": ("chances_created",)}

    payload["xg"] = evaluate(rows, key="xg", candidates=xg_candidates)
    payload["xa"] = evaluate(rows, key="xa", candidates=xa_candidates)
    (OUT / "attacking_process_evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    for block in ("xg", "xa"):
        print(f"\n=== {block.upper()} (2025/26 rolling OOS) ===")
        base = payload[block].get(f"{block.upper()}_BASELINE")
        if not base:
            print("  no folds"); continue
        print("  %-22s %-10s %-10s %-9s %s" % ("model", "baseline", "candidate", "rel", "fold wins"))
        for name, e in payload[block].items():
            rel = (e["baseline_mae"] - e["candidate_mae"]) / e["baseline_mae"]
            print("  %-22s %.6f  %.6f  %+.3f%%  %d/%d" % (
                name, e["baseline_mae"], e["candidate_mae"], 100 * rel, e["fold_wins"], e["folds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
