# FPL STRATEGY V1 FINAL

**Status:** FINAL

**Purpose.** This is the canonical manager-level decision framework for the 2026/27 FPL season. It governs decisions that sit ABOVE FPL Brain, Scout Operations, and the AI Scout. It does not change the responsibility boundaries of either system, and it does not turn FPL Brain into an automated manager. The human manager remains responsible for every final decision: transfers, rolling free transfers, hits, price-change reactions, captaincy, vice-captaincy, squad structure, Wildcard, Triple Captain, Bench Boost, Free Hit, chip timing, and mini-league play.

In the project's provenance vocabulary, this document is the governance layer for the `[MANUAL]` tier: it consumes `[FACT]`, `[SCOUTING]`, and `[DERIVED]` information produced by the systems below it and decides.

---

## 1. SYSTEM RESPONSIBILITY BOUNDARIES

### FPL Brain — FACT + DERIVED

Responsible for structured information: squad, prices, ownership, fixtures, official player data, player snapshots, manager state, transfers, chip state, historical results, derived transparent metrics, and the log of previous decisions. FPL Brain does not autonomously choose transfers, captains, or chips. It stores manager decisions (`decisions`, `strategy`, `watchlist`) as `[MANUAL]` records; it never makes them.

### AI Scout — SCOUTING

Responsible for contextual uncertainty: start probability, expected minutes, tactical role, role changes, penalties, free kicks, corners, fitness/readiness, manager comments, competition for position, rotation, role security, transfer risk, and congestion — all imported as `[SCOUTING]` observations with confidence, evidence, and timing under `LUNA_SCOUT_PROTOCOL_V1.md`. The scout does NOT recommend transfers, captains, chips, or rankings. It answers _"what does the evidence say?"_, never _"what should we do?"_.

### Manager Strategy — MANUAL DECISION

The manager combines FACT + SCOUTING + DERIVED + strategic context to decide: start / bench, buy / sell / hold, roll, hits, captain / vice, chips, and squad restructuring. Final responsibility remains with the human manager. Every framework rule below is a discipline for that judgement, not a replacement for it.

---

## 2. PRIMARY OBJECTIVE

The default objective is to **MAXIMISE EXPECTED SEASON POINTS**.

Do not optimise for: last week's points; ownership alone; team value alone; avoiding red arrows; accumulating free transfers for its own sake; or finding differentials for their own sake.

**Early and middle season:** primarily maximise expected FPL points.

**Late season:** the objective may change when the specific goal is winning a mini-league. Then the probability of achieving the target outcome may matter more than maximum raw expected points (section 18).

---

## 3. TRANSFER DECISION RULE

Every transfer must compete against **DO NOTHING**.

Do not ask only: _"Is Player B better than Player A?"_
Ask: _"Is Player B worth transferring in NOW compared with every alternative, including holding and rolling?"_

Conceptual evaluation:

```
TRANSFER NET VALUE =
    expected points gained over intended holding period
  + captaincy value
  + chip-enabling value
  + structural value
  + affordability-route value
  - hit cost
  - marginal value of the free transfer consumed
  - information risk from acting early
  - future cleanup cost
```

This is a decision framework, NOT an opaque numerical model. Do not invent false precision: the scouting inputs are confidence-banded judgement (`95/85/70/50/30/15/5`), and the evaluation inherits that honesty.

---

## 4. ACTUAL HOLDING PERIOD

Do not rigidly evaluate every transfer across exactly five Gameweeks. Use the **realistic intended holding period**.

Examples:

- one-week punt before a Wildcard → 1 GW;
- temporary injury replacement → until the expected return;
- fixture-run transfer → approximately 3–6 GW;
- structural premium or goalkeeper → potentially much longer.

Keep GW+1, GW+3, and GW+5 views as useful reference horizons where available (they match the report's fixture horizons and `role_security_5gw`), but the **intended holding period governs the actual decision**.

---

## 5. FREE-TRANSFER OPTION VALUE

A free transfer is **a perishable option, not a trophy**.

Rolling is correct only when the best available move does not beat the marginal value of retaining the transfer. The value of a saved FT depends on:

- the number already banked;
- upcoming uncertainty and injuries;
- expected fixture swings;
- squad fragility;
- upcoming chip plans;
- potential restructuring needs;
- information expected before the next deadline.

Do NOT hold a target of automatically accumulating 3–5 FTs. Banking several transfers can be powerful, but accumulation itself is not the objective.

As the bank approaches the maximum allowed number of FTs, the marginal value of another saved FT falls. **At the maximum bank, the threshold for spending one on a genuine positive-EV move should be low**, because failing to spend may waste the incoming transfer.

---

## 6. HIT POLICY

Do not use the vague rule "never take hits". Also do not take hits simply because an incoming player has a better fixture. A hit must be evaluated net of its cost:

```
NET HIT VALUE =
    expected gain versus the TRUE baseline
  + structural / captaincy / chip value
  - 4 points
  - future cleanup cost
  - lost information value
```

The **TRUE baseline** may be: the outgoing player; the first bench player; the autosub; an alternative transfer route; or doing nothing. Comparing against the outgoing player's name alone is not a baseline.

Hits become more defensible when:

- the outgoing player has near-zero expected minutes;
- bench cover is weak or unavailable;
- the move creates an elite captaincy route;
- the move solves several future problems at once;
- the improvement persists across multiple GWs;
- the role/minutes evidence is strong (`[SCOUTING]` confidence high, not hoped);
- the move materially helps a planned chip.

With up to five bankable FTs, discretionary hits should generally be less common — but never prohibited by ideology.

---

## 7. PRICE CHANGES AND AFFORDABILITY CLIFFS

Price movement matters only insofar as it improves or damages **future expected points**. Do not chase team value for its own sake.

Distinguish:

- **COSMETIC PRICE MOVEMENT** — displayed team value drifts, no planned route is affected;
- **AFFORDABILITY CLIFF** — a £0.1m change materially destroys an important planned route.

Before making an early transfer because of a likely rise or fall, ask:

1. Does waiting likely make an important route impossible?
2. Is the change in actual selling value, or only displayed team value?
3. Is the target's role/minutes/fitness evidence already strong?
4. Is important team news still expected?
5. Is the money required for a real squad plan?

If there is no meaningful affordability cliff, the value of waiting for information may exceed £0.1m. If the route would genuinely die **and** the football evidence is already strong, early action can be correct.

---

## 8. VALUE OF INFORMATION

Waiting has value only when future information can plausibly **change the decision**.

Useful future information includes: press conferences, training reports, injuries, registration, tactical information, cup minutes, transfers, suspensions, and credible lineup information — the same evidence classes the Scout Protocol grades.

Do not wait simply because waiting feels cautious. If little important information is expected and a move has clear positive value, act.

---

## 9. CAPTAINCY — EXPECTED ARMBAND VALUE

Captaincy should maximise **EXPECTED ARMBAND VALUE**. Explicitly consider: start probability, expected minutes, expected points if starting, cameo probability, expected cameo contribution, complete no-show probability, vice-captain fallback, penalties and set pieces, attacking role, opponent, team attacking strength, congestion, and fitness/load uncertainty.

**A bench cameo and a complete no-show are strategically different.** If the captain completely misses the match, the vice-captain can activate. If the captain comes on for a short cameo, the vice-captain does NOT replace him. Therefore two players with the same start probability can have very different captaincy value.

Conceptual accounting:

```
Expected captaincy bonus ≈
    P(captain appears) × expected captain points conditional on appearance
  + P(captain completely misses out AND vice appears)
    × expected vice-captain points
```

Do NOT use the erroneous formulation that subtracts vice-captain value from the captain's line. This is conceptual expected-value accounting, not a requirement for fake numerical precision.

---

## 10. VICE-CAPTAINCY

The vice-captain should preferably have: high expected minutes, low no-show probability, strong expected points, and **low SHARED NON-APPEARANCE risk with the captain**.

The important concern is not scoring correlation. The concern is whether captain and vice could **both fail to appear** due to: postponement, weather, a same-team rotation event, shared injury/illness context, or any other common non-appearance risk.

---

## 11. OWNERSHIP AND CAPTAINCY

**Early/mid-season:** ownership should NOT determine captaincy. Choose primarily on expected armband value. Do not use "safe captain" unless there is a mathematically meaningful objective behind it.

**Late season:** ownership and effective ownership may matter when trying specifically to win a mini-league. Relevant **rival** ownership and **rival captaincy** matter more than generic global ownership.

---

## 12. SQUAD STRUCTURE GUARDRAILS

### Captaincy routes

Avoid structures that make elite captaincy options unnecessarily difficult to reach.

### Price-point flexibility

Avoid dead-end structures that require multiple transfers merely to move between common player price brackets.

### Playable first bench

The first substitute should ideally offer credible minutes, so a surprise benching does not destroy the Gameweek.

### Aggregate minutes-risk budget

Do not evaluate minutes risk only player-by-player. One 70% starter may be acceptable; five uncertain starters may create excessive squad-wide risk. (The report's RISKS section surfaces the individual low-confidence notes; the manager judges the aggregate.)

### Bench spending

Do not overfund substitutes unless deliberately preparing a Bench Boost or there is another clear strategic reason.

---

## 13. WILDCARD

The Wildcard is both a **repair chip** and an **opportunity chip**.

Do NOT Wildcard because: last Gameweek was bad; the rank fell; several players blanked.

Evaluate the **expected value of the Wildcard squad** against the **best non-Wildcard path**.

Strong triggers can include:

- multiple role/minutes failures;
- many injuries;
- a structural price-point problem;
- a major multi-team fixture swing;
- weak captaincy access;
- 6–8 genuinely undesirable squad slots;
- a strong Bench Boost setup;
- chip expiration approaching;
- banked FTs still cannot efficiently repair the squad.

Bankable FTs raise the Wildcard threshold, but do not make the Wildcard taboo.

---

## 14. TRIPLE CAPTAIN

Triple Captain value is effectively **one additional copy of the captain's score**.

Do not automatically save it for "any Double Gameweek". Do not automatically reject a Double Gameweek just to be contrarian. Compare the **expected TC opportunity now** against **realistic remaining TC opportunities**.

Ideal characteristics: an elite attacker, exceptional expected points, penalties, an excellent role, high minutes confidence, favourable fixture(s), two strong expected starts if it is a DGW, and low rotation/injury concern.

An exceptional SGW opportunity can beat a mediocre DGW. An elite attacker with two strong DGW starts will usually be difficult for an SGW to beat.

---

## 15. BENCH BOOST

Evaluate **NET Bench Boost value**, not simply "how many points will my bench score?".

```
NET BB VALUE ≈
    expected bench points
  - hits / preparation cost
  - long-term XI weakening
  - cost of carrying unnecessary bench money
  - future cleanup cost
```

Strong setup: 15 credible players, strong minutes expectation, useful fixtures, few or no forced hits, no damage to XI quality, and no harmful long-term bench investment.

Do not take a −4 merely to improve one Bench Boost substitute unless the combined BB-week and future expected value clearly justifies it.

---

## 16. FREE HIT

Free Hit is useful for **TEMPORARY DAMAGE** and **TEMPORARY OPPORTUNITY**.

Examples: a major Blank GW; a strong Double GW; an unusual one-week fixture distortion; a late-season tactical mini-league opportunity.

Free Hit is weaker when: the permanent squad is fundamentally bad; banked FTs solve the problem cheaply; or a stronger future FH opportunity likely exists.

Use it to optimise one unusual Gameweek **without damaging a good permanent squad**.

---

## 17. CHIP OPPORTUNITY CALENDAR

Do NOT lock exact chip weeks months in advance. Maintain a **rolling opportunity assessment**.

For each chip, grade candidate windows as **WEAK / MEDIUM / STRONG / EXCEPTIONAL**, and add **OPPORTUNITY DECAY**. Ask:

- how good is the current opportunity?
- how good are realistic future alternatives?
- how many opportunities remain?
- when does the chip expire?
- what preparation is required?
- what is the transfer/hit cost?
- does waiting genuinely preserve something valuable?

As realistic future windows disappear or expiry approaches, the threshold for using the chip should fall. **Do not preserve optionality into forced mediocrity.**

---

## 18. MINI-LEAGUE GAME THEORY

**Early/mid-season:** maximise expected points. Do not chase low ownership for its own sake.

**Late season:** the objective may change from maximum expected FPL points to **maximum probability of achieving the target mini-league outcome**.

**If leading:**

- covering a rival captain can be rational when the EV sacrifice is small;
- do not copy objectively weak rival decisions;
- protect primarily against high-impact correlated risks such as captaincy and chips.

**If chasing:**

- prefer higher variance where EV is similar;
- accept some EV sacrifice only when the gap and time remaining justify it;
- concentrate risk into powerful levers such as captaincy and chips;
- do not fill the team with inferior "differentials".

Variance is a tool only after defining: the league target, the points gap, the Gameweeks remaining, the rival overlap, and the captaincy/chip state.

---

## 19. DECISION JOURNAL

For significant decisions, record BEFORE the outcome is known:

- the decision;
- expected points or expected range;
- intended holding period;
- confidence;
- main uncertainty;
- second-best alternative;
- value of waiting;
- transfer cost;
- price/affordability issue;
- what new evidence would change the decision.

Later, record: the actual outcome; whether assumptions were correct; process quality; and the calibration lesson.

**Never rewrite the original expectation after seeing the result.** A good decision can produce a bad outcome; a bad decision can produce a good outcome. Judge PROCESS separately from OUTCOME. (The `decisions` table is the natural home for these entries; its assumptions/invalidators/review fields match this section.)

---

## 20. WEEKLY MANAGER REVIEW

Practical weekly checklist, in order:

1. Sync FPL Brain facts (`fetch_fpl.py`, `sync_manager.py`).
2. Review manager state and the FT bank.
3. Review injuries / unavailable players.
4. Generate the Scout Operations brief (`build_scout_brief.py`).
5. Run Scout research where needed.
6. Import validated scouting evidence (`validate_scouting.py`, then `import_scouting.py`).
7. Rebuild the FPL Brain report (`build_report.py`).
8. Review the starting XI and bench.
9. Review captain / vice.
10. Review transfer candidates versus DO NOTHING.
11. Assess marginal FT value / whether to roll.
12. Check hit economics.
13. Check price-change affordability cliffs.
14. Review squad structure.
15. Review current and future chip opportunities.
16. Consider mini-league context only where strategically relevant.
17. Record major decision expectations.
18. Lock the team.
19. Post-GW: review process separately from result.

---

## 21. DECISION OUTPUT TEMPLATE

Canonical manager-decision template:

```
GAMEWEEK:

CURRENT STATE
- Free transfers:
- Bank:
- Team value:
- Key injuries:
- Chips available:

LINEUP
- Starting XI:
- Bench order:

CAPTAINCY
- Captain:
- Vice:
- Captain start probability:
- Cameo risk:
- No-show / VC protection:
- Reason:

TRANSFERS
- Proposed action:
- Alternative: DO NOTHING
- Intended holding period:
- Expected gain/range:
- FT option cost:
- Hit:
- Information value:
- Affordability cliff:
- Structural effect:

CHIPS
- Current chip opportunity:
- Best realistic future opportunity:
- Expiry pressure:
- Decision:

MINI-LEAGUE CONTEXT
- Only when relevant.

FINAL CALL:
- BUY / SELL / HOLD / ROLL / HIT / WAIT
- CAPTAIN:
- CHIP:
- CONFIDENCE:

DECISION-JOURNAL NOTE:
- Main uncertainty:
- Second-best option:
- Evidence that would change decision:
```

Do not require every field when it is irrelevant.

---

## 22. ANTI-BIAS RULES

- No outcome bias.
- No recency chasing.
- No FOMO.
- No price-rise chasing for vanity.
- No differential for its own sake.
- No ownership-driven early-season captaincy.
- No Wildcard because of one bad score.
- No arbitrary chip procrastination.
- No free-transfer hoarding ideology.
- No transfer simply because a player is objectively good — he must beat DO NOTHING for THIS squad NOW.
- No false precision.
- Uncertainty may remain unresolved; acting despite residual uncertainty is often correct.

---

## 23. WHAT THIS DOCUMENT DOES NOT DO

`FPL_STRATEGY_V1_FINAL.md` does NOT:

- alter FPL Brain;
- alter Scout Operations;
- alter the Scout Protocol;
- create transfer recommendations automatically;
- create a player-ranking algorithm;
- create a captain optimiser;
- create a chip optimiser;
- create a prediction model;
- create a new database table;
- create automation;
- create a dashboard.

It is a manager decision framework.

---

## 24. CURRENT PLAYER PICKS ARE NOT CANONICAL STRATEGY

Do not encode the current GW1 squad, any current captaincy decision, or any specific current player choice as permanent strategy. Specific players and Gameweeks change. The strategy governs **HOW** those decisions are made, not **WHO** they pick.

---

# STRATEGY FREEZE

FPL Strategy V1 is frozen for normal weekly use.

It should NOT be rewritten because: one captain blanks; a transfer fails; a benched player hauls; the rank falls; or a chip produces a poor outcome.

Changes require evidence of a systematic process flaw, an FPL rule change, or a materially better decision principle. Normal learning occurs through the decision journal and calibration — not continual architecture redesign.
