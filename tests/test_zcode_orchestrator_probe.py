"""Focused probe tests for the disposable ZCode/Agent-Orchestrator integration lab.

Not FPL product logic: this module only asserts the contract of
``integration_lab.pricing.calculate_total``.
"""

from __future__ import annotations

import pytest

from integration_lab.pricing import calculate_total


# 1 -----------------------------------------------------------------


def test_ordinary_valid_case_returns_exact_total():
    assert calculate_total(10.0, 3, 10) == 27.0


# 2 -----------------------------------------------------------------


def test_zero_discount_returns_full_total():
    assert calculate_total(25.0, 4, 0) == 100.0


# 3 -----------------------------------------------------------------


def test_hundred_percent_discount_returns_exactly_zero():
    total = calculate_total(19.99, 7, 100)
    assert total == 0.0
    assert isinstance(total, float)


# 4 -----------------------------------------------------------------


def test_decimal_price_returns_exact_total():
    assert calculate_total(12.5, 3, 10) == 33.75


# 5 -----------------------------------------------------------------


def test_zero_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(0, 3, 10)
    with pytest.raises(ValueError):
        calculate_total(0.0, 3, 10)


# 6 -----------------------------------------------------------------


def test_negative_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(-5.0, 3, 10)


# 7 -----------------------------------------------------------------


def test_zero_quantity_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, 0, 10)


# 8 -----------------------------------------------------------------


def test_negative_quantity_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, -2, 10)


# 9 -----------------------------------------------------------------


def test_bool_quantity_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, True, 10)
    with pytest.raises(ValueError):
        calculate_total(10.0, False, 10)


# 10 ----------------------------------------------------------------


def test_non_integer_quantity_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, 2.5, 10)
    with pytest.raises(ValueError):
        calculate_total(10.0, 3.0, 10)


# 11 ----------------------------------------------------------------


def test_discount_below_zero_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, -1)


# 12 ----------------------------------------------------------------


def test_discount_above_hundred_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, 101)
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, 100.1)


# 13 ----------------------------------------------------------------


def test_bool_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(True, 3, 10)


# 14 ----------------------------------------------------------------


def test_bool_discount_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, True)
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, False)


# 15 ----------------------------------------------------------------


def test_nan_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(float("nan"), 3, 10)


# 16 ----------------------------------------------------------------


def test_infinite_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(float("inf"), 3, 10)
    with pytest.raises(ValueError):
        calculate_total(float("-inf"), 3, 10)


# 17 ----------------------------------------------------------------


def test_nan_discount_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, float("nan"))


# 18 ----------------------------------------------------------------


def test_infinite_discount_is_rejected():
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, float("inf"))
    with pytest.raises(ValueError):
        calculate_total(10.0, 3, float("-inf"))


# 19 ----------------------------------------------------------------


def test_overflowing_arithmetic_never_returns_a_non_finite_total():
    with pytest.raises(ValueError):
        calculate_total(1e308, 10, 0)


# 20 ----------------------------------------------------------------


def test_extremely_large_integer_price_is_rejected():
    huge_price = 10**400
    with pytest.raises(ValueError):
        calculate_total(huge_price, 3, 10)
    with pytest.raises(ValueError):
        calculate_total(huge_price, 3, 0)


# 21 ----------------------------------------------------------------


def test_extremely_large_integer_quantity_is_rejected():
    huge_quantity = 10**400
    with pytest.raises(ValueError):
        calculate_total(10.0, huge_quantity, 10)
    with pytest.raises(ValueError):
        calculate_total(10.0, huge_quantity, 0)
