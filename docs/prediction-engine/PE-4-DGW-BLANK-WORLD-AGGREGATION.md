# Prediction Engine V1 — PE-4 DGW / Blank World Aggregation

## Objective

Certify that fixture-level simulated football worlds aggregate correctly into FPL event worlds for:

- SGW
- DGW
- blank Gameweeks

without changing football equations or RNG.

## Fixture atomicity

Fixtures remain independent football simulation units.

## Event world contract

For player P and simulation index w:

```text
event_core(P,w)
=
sum(
    fixture_core(P,f,w)
    for all target-event fixtures f containing P
)

event_minutes(P,w)
=
sum(
    fixture_minutes(P,f,w)
    for all target-event fixtures f containing P
)
```

### SGW

one fixture -> exact event-series identity

### DGW

multiple fixtures -> additive at SAME world index

### Blank

captured player has explicit zero core series and zero minutes series.

A blank is NOT:

- missing key
- `None`
- fake fixture
- average
- last-fixture value

## Manager semantics

A player appears in an event if aggregated event minutes > 0.

For DGW:

- 0 + positive -> appeared
- positive + 0 -> appeared
- 0 + 0 -> absent
- positive + positive -> appeared

Do NOT cap event minutes at 90.

Captaincy is event-level.

Autosubs consume aggregated event appearance.

Bench Boost consumes event aggregate.

Free Hit consumes event aggregate.

## Expected bonus

Existing soft bonus path remains authoritative in PE-4.

DGW expected soft bonus is summed across fixtures.

Do NOT promote PE-3 structural bonus here.

## Role actionability

Do not numerically aggregate it.
Use existing event/player certified state.

## Out of scope

- PE-1
- PE-2
- PE-3
- minutes refinement
- team model refinement
- player-rate refinement
- calibration
- certification promotion
- PE-5 outcome work
- chip redesign
- transfer redesign

## Version/RNG

`MONTE_CARLO_MODEL_VERSION` remains `mc_v1.3.0`.

**NEW RNG: NONE**

Config hash semantics unchanged.

A model/version bump requires Product Owner escalation.

## Test-first policy

Current code appears already designed to aggregate fixtures by adding each fixture into shared `world_core` / `world_minutes` arrays at the same simulation index.

Therefore:

**WRITE DISCRIMINATING TESTS FIRST.**

Do not rewrite working aggregation unless a failing contract test proves a real defect.

## Required hard cases

1. SGW exact core identity
2. SGW exact minutes identity
3. DGW world-by-world core sum
4. DGW world-by-world minutes sum
5. blank explicit core zeros
6. blank explicit minutes zeros
7. blank key present
8. missing key rejected
9. DGW 0 + positive appearance
10. DGW positive + 0 appearance
11. DGW 0 + 0 absence
12. minutes >90 supported
13. no averaging
14. same-index world coherence
15. mixed players with 2 / 1 / 0 fixtures
16. player-specific schedules
17. captain DGW event total
18. captain appears in only one fixture -> no vice fallback
19. captain zero in all fixtures -> existing vice fallback
20. autosub DGW appearance
21. blank starter autosub
22. Bench Boost DGW sum
23. Bench Boost blank
24. Free Hit DGW
25. Free Hit blank
26. another-event fixture excluded
27. duplicate fixture identity not double-counted
28. soft DGW bonus summed
29. blank soft bonus zero
30. role actionability unchanged
31. deterministic repeated run

## Terminal state

`READY_FOR_MERGE`

No merge.

No PE-5.
