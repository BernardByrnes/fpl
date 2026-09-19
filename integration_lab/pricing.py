"""Disposable ZCode/Agent-Orchestrator integration lab: trivial price arithmetic.

This module is deliberately not FPL product logic.  It exists only so the
orchestrator has a self-contained deliverable to build and verify.
"""

from __future__ import annotations


def calculate_total(price, quantity, discount_percent):
    """Return ``price * quantity`` less ``discount_percent``, rounded to 2dp.

    Raises ``ValueError`` for any invalid input; a rejected input never yields a
    substitute total.
    """
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise ValueError(f"price must be an int or float, got {type(price).__name__}")
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
    if not 0 <= discount_percent <= 100:
        raise ValueError(
            f"discount_percent must be between 0 and 100 inclusive, got {discount_percent!r}"
        )

    return round(price * quantity * (1 - discount_percent / 100), 2)
