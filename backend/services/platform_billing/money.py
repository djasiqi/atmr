"""Arrondi monétaire CHF plateforme (multiples de 0,05)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

_TWO = Decimal("0.01")
_FIVE_CENTS = Decimal("0.05")


def money_round_chf(value: Decimal) -> Decimal:
    """Arrondit un montant au multiple de 5 centimes (HALF_UP), puis 2 décimales.

    Exemples :
        - 1687.14 → 1687.15
        - 10.32 → 10.30
        - 10.33 → 10.35
    """
    amount = Decimal(value)
    rounded = (amount / _FIVE_CENTS).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ) * _FIVE_CENTS
    return rounded.quantize(_TWO, rounding=ROUND_HALF_UP)
