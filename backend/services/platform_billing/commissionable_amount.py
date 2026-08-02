"""Résolution du montant commissionnable (Decimal, jamais float natif)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any


class CommissionAmountSource(str, Enum):
    FINAL_TRANSPORT_PRICE = "FINAL_TRANSPORT_PRICE"
    PRICE_AMOUNT = "PRICE_AMOUNT"
    CANCELLATION_FEE = "CANCELLATION_FEE"
    LEGACY_BOOKING_AMOUNT = "LEGACY_BOOKING_AMOUNT"
    NONE = "NONE"


class AmountConfidence(str, Enum):
    CERTAIN = "CERTAIN"
    LEGACY = "LEGACY"
    MISSING = "MISSING"


@dataclass(frozen=True)
class CommissionableAmount:
    amount: Decimal | None
    source: CommissionAmountSource
    confidence: AmountConfidence
    reason: str | None = None


def _positive_decimal(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    try:
        d = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if d <= 0:
        return None
    return d


def resolve_commissionable_amount(
    booking: Any,
    *,
    cancellation_policy: str = "exclude",
) -> CommissionableAmount:
    """Ordre : prix final verrouillé → price_amount → fee annulation → amount Float legacy."""
    status = getattr(booking, "status", None)
    status_v = getattr(status, "value", status)

    # Annulation
    if status_v in ("CANCELED", "CANCELLED"):
        if cancellation_policy == "exclude":
            return CommissionableAmount(
                None,
                CommissionAmountSource.NONE,
                AmountConfidence.MISSING,
                "CANCELLED_EXCLUDED",
            )
        if cancellation_policy == "on_cancellation_fees":
            fee = _positive_decimal(getattr(booking, "cancellation_fee_amount", None))
            if fee is None:
                return CommissionableAmount(
                    None,
                    CommissionAmountSource.NONE,
                    AmountConfidence.MISSING,
                    "CANCELLATION_FEE_NOT_AVAILABLE",
                )
            return CommissionableAmount(
                fee,
                CommissionAmountSource.CANCELLATION_FEE,
                AmountConfidence.CERTAIN,
            )
        # on_billed_amount : continue vers montants normaux

    # Montant final éventuel (facture client liée / attribut métier)
    for attr in ("final_billable_amount", "locked_price_amount"):
        d = _positive_decimal(getattr(booking, attr, None))
        if d is not None:
            return CommissionableAmount(
                d,
                CommissionAmountSource.FINAL_TRANSPORT_PRICE,
                AmountConfidence.CERTAIN,
            )

    price_amount = _positive_decimal(getattr(booking, "price_amount", None))
    if price_amount is not None:
        return CommissionableAmount(
            price_amount,
            CommissionAmountSource.PRICE_AMOUNT,
            AmountConfidence.CERTAIN,
        )

    legacy = _positive_decimal(getattr(booking, "amount", None))
    if legacy is not None:
        return CommissionableAmount(
            legacy,
            CommissionAmountSource.LEGACY_BOOKING_AMOUNT,
            AmountConfidence.LEGACY,
            "LEGACY_BOOKING_AMOUNT",
        )

    return CommissionableAmount(
        None,
        CommissionAmountSource.NONE,
        AmountConfidence.MISSING,
        "FINAL_AMOUNT_NOT_AVAILABLE",
    )
