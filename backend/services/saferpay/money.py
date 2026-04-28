"""Montants Saferpay Payment Page (Value = centimes CHF en chaîne)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def _to_decimal_amount(amount: Decimal | float | int | str) -> Decimal:
    if isinstance(amount, Decimal):
        return amount
    return Decimal(str(amount))


def chf_amount_to_saferpay_value_str(amount: Decimal | float | int | str) -> str:
    d = _to_decimal_amount(amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return str(int(d * 100))
