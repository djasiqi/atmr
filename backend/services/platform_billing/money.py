"""Arrondi monétaire half-up 2 décimales (CHF)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

_TWO = Decimal("0.01")


def money_round_chf(value: Decimal) -> Decimal:
    return value.quantize(_TWO, rounding=ROUND_HALF_UP)
