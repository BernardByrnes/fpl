"""Disposable ZCode/Agent-Orchestrator integration lab: trivial price arithmetic.

This module is deliberately not FPL product logic.  It exists only so the
orchestrator has a self-contained deliverable to build and verify.
"""

from __future__ import annotations

import math


def calculate_total(price, quantity, discount_percent):
    """Return ``price * quantity`` less ``discount_percent``, rounded to 2dp.

    Raises ``ValueError`` for any invalid input; a rejected input never yields a
    substitute total, and a finite-looking input never yields a non-finite one.
    """
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise ValueError(f"price must be an int or float, got {type(price).__name__}")
    # Ints are finite by construction, and math.isfinite() raises OverflowError
    # on an int too large to convert to a float, so only floats are tested.
    if isinstance(price, float) and not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")
    if price <= 0:
        raise ValueError(f"price must be greater than zero, got {price!r}")

    if isinstance(quantity, bool) or not isinstance(quantity, int):
        raise ValueError(f"quantity must be an int, got {type(quantity).__name__}")
    if quantity <= 0:
        raise ValueError(f"quantity must be greater than zero, got {quantity!r}")

    if isinstance(discount_percent, bool) or not isinstance(discount_percent, (int, float)):
        raise ValueError(
            f"discount_percent must be an int or float, got {type(discount_percent).__name__}"
        )
    if isinstance(discount_percent, float) and not math.isfinite(discount_percent):
        raise ValueError(f"discount_percent must be finite, got {discount_percent!r}")
    if not 0 <= discount_percent <= 100:
        raise ValueError(
            f"discount_percent must be between 0 and 100 inclusive, got {discount_percent!r}"
        )

    try:
        total = price * quantity * (1 - discount_percent / 100)
        is_finite = math.isfinite(total)
    except OverflowError as exc:
        # An exact integer product can be too large to convert to a float; that
        # input has no representable total, so reject it like a non-finite one
        # rather than letting OverflowError escape past the contract.
        raise ValueError("total is too large to represent as a float") from exc
    if not is_finite:
        raise ValueError(f"total must be finite, got {total!r}")
    return round(total, 2)
