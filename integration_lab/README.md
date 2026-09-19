# integration_lab

> THIS DIRECTORY IS A DISPOSABLE ZCODE/AGENT-ORCHESTRATOR INTEGRATION LAB.
> IT IS NOT FPL PRODUCT LOGIC.

This package exists only as a self-contained fixture that exercises the Agent
Orchestrator end-to-end flow (implement, test, commit, verify). It contains no
FPL scoring, prediction, or decision logic and carries no authority from the
project's product specification.

## Production code must not import this

Nothing in the FPL product — `fpl_brain/`, `scripts/`, or any other shipping
path — may import `integration_lab`, directly or indirectly. It depends on
nothing from the product and the product must depend on nothing here. This
directory can be deleted at any time without affecting FPL behaviour.

## Contents

- `pricing.py` — `calculate_total(price, quantity, discount_percent)`, a trivial
  arithmetic function returning the total rounded to two decimal places, and
  raising `ValueError` on any invalid input.
- `__init__.py` — package marker.

## Running the focused tests

From the repository root:

```
python -m pytest tests/test_zcode_orchestrator_probe.py -q
```

The probe lives at `tests/test_zcode_orchestrator_probe.py` and depends only on
`pytest` and the Python standard library.
