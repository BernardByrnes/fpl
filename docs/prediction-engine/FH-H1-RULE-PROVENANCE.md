# Free Hit H1 hit-rule provenance — 2026/27

**Disposition: authoritative source exists; current-season FH H1 zero-hit provenance is closed.** This note binds the source pin below to the current implementation. It does not change runtime behavior.

## Authoritative rule source

The Premier League's official **FPL Rules Copilot**, published 18 May 2026, defines Free Hit as: “Make unlimited free transfers for a single Gameweek.” Its transfer section says each additional transfer in a Gameweek deducts four points. Read together, the chip-specific rule makes the transfers in the active Free Hit Gameweek free of transfer-hit cost.

- Rules page: <https://www.premierleague.com/en/news/4661029>
- Captured page identity: article ID `4661029`, title `FPL Rules Copilot`, in-page publication date `18 May 2026`.
- Captured response body: `K:/zcode-orchestrator-bridge/evidence/fpl_fhwc_fh_h1_provenance_disposition_20261009/official-source-pins/premier-league-fpl-rules-copilot-2026-05-18.html`
- Captured bytes: `179812`; SHA-256: `D1B2707E52B048C6A24EB12A7BBD3F2689BE95DFEC1E988E01BD238B985A8F65`.

Season applicability is corroborated by the Premier League's explicit 2026/27 chips article, published 20 July 2026, which repeats the same Free Hit rule and identifies the season:

- Season article: <https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627>
- Captured page identity: article ID `4679879`, title `What’s happening with FPL chips in 2026/27`, in-page publication date `20 Jul 2026`.
- Captured response body: `K:/zcode-orchestrator-bridge/evidence/fpl_fhwc_fh_h1_provenance_disposition_20261009/official-source-pins/premier-league-fpl-chips-2026-27-2026-07-20.html`
- Captured bytes: `92469`; SHA-256: `C4DF26BA7467D98FBBDEBF77D2276ED12F8BCBD0967CA0864205D4E7F34A4A23`.

Neither page exposes a separate rules-version number. The official publisher, article ID, publication date, URL and captured-byte hash identify the pinned versions. The pins were retrieved on 9 October 2026. The disposition record is `K:/zcode-orchestrator-bridge/evidence/fpl_fhwc_fh_h1_provenance_disposition_20261009/FINAL_DISPOSITION.md`.

## Implementation binding and boundary

`fpl_brain/chip_horizon_evidence_assembly.py::verify_future_action_hit_cost_witnesses` constructs the FH PLAY H1 witness row as `hit_points: 0` (line 1772). This is the implementation expression of the official active-chip rule above. The current code does **not** resolve that H1 value from `SeasonRules` or carry the page hash in the runtime evidence; the binding is the explicit, reviewed documentation mapping to the source pin in this file.

The rest of the acceptance gate remains source-replayed. `_require_zero_hit_arms` requires exact integer zero for both complete PLAY and SAVE schedules. A nonzero replay-derived hit is refused before first retention, and publication recovery/load re-runs the witness verifier. Capture and maturation load through the same verified forecast-origin path, so a paid-hit schedule cannot enter the accepted FH evidence lifecycle. The official realization path is gross-only; paid-hit FH realization/calibration remains unsupported.

The literal H1 value does not automatically detect a future official-rule change or a noncanonical producer that tries to attach a paid hit to the active FH event. That is a separate code-robustness residual noted by the prior review. The pinned source establishes that the current 2026/27 rule supports zero for canonical FH H1; it is not a claim of automatic future-rule drift detection.

FT/reset semantics are separate. `SeasonRules.free_hit_ft_rule` concerns saved free-transfer preservation, while restoration of the pre-chip squad at the next deadline is another Free Hit rule. Neither is the source for the H1 hit-cost value.

## Acceptance boundary

- Supported: zero transfer-hit cost for canonical moves made during the active 2026/27 Free Hit Gameweek.
- Unsupported: any paid-hit FH schedule reaching realized scoring or calibration. Nonzero source-replayed schedule rows fail closed before issuance; current evidence/calibration does not support netting paid hits from gross realized points.
- No autonomous authority is inferred from the positive fixture or passing lifecycle test; rule authority is the pinned Premier League source above.
- This is a documentation-only source binding. No production code or test was changed or run for this provenance disposition.
