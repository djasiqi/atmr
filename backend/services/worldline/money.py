"""Conversion montants pour l'API Worldline (minor units)."""

from __future__ import annotations

from decimal import Decimal


def chf_amount_to_cents(amount: float) -> int:
    d = Decimal(str(amount)).quantize(Decimal("0.01"))
    return int(d * 100)
