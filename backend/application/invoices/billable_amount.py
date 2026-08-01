"""Montant HT facturable canonique pour un booking (preview / generate / totaux)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from infrastructure.invoices.invoice_calculator import round_to_5_cents

_TWO = Decimal("0.01")


@dataclass(frozen=True)
class BillableAmount:
    amount_ht: Decimal
    source: str
    cancellation_fee_applied: bool
    catalog_amount_ht: Decimal | None


def calculate_billable_booking_amount(
    booking: Any,
    *,
    billing_settings: Any = None,
    override: dict[str, Any] | None = None,
) -> BillableAmount:
    """Calcule le HT facturable (annulation = frais, livraison = prix fixe settings)."""
    mission_type = getattr(booking, "mission_type", None) or "patient_transport"
    catalog: Decimal | None = None
    cancellation_fee_applied = False
    source = "booking.amount"

    if mission_type == "material_delivery":
        fp = None
        if billing_settings is not None:
            fp = getattr(billing_settings, "material_delivery_price_fixed", None)
        if fp is None or fp <= 0:
            return BillableAmount(
                amount_ht=Decimal("0.00"),
                source="material_delivery_unconfigured",
                cancellation_fee_applied=False,
                catalog_amount_ht=None,
            )
        amount = Decimal(str(fp)).quantize(_TWO)
        source = "material_delivery_fixed"
        return BillableAmount(
            amount_ht=round_to_5_cents(amount),
            source=source,
            cancellation_fee_applied=False,
            catalog_amount_ht=None,
        )

    catalog = Decimal(str(getattr(booking, "amount", None) or 0)).quantize(_TWO)
    amount = catalog

    if (
        str(getattr(booking, "status", "") or "").upper() == "CANCELED"
        and getattr(booking, "cancellation_fee_amount", None) is not None
    ):
        amount = Decimal(str(booking.cancellation_fee_amount)).quantize(_TWO)
        cancellation_fee_applied = True
        source = "cancellation_fee_amount"

    if override and override.get("amount") is not None:
        try:
            amount = Decimal(str(override["amount"])).quantize(_TWO)
            source = "override.amount"
        except Exception:
            pass

    return BillableAmount(
        amount_ht=round_to_5_cents(amount),
        source=source,
        cancellation_fee_applied=cancellation_fee_applied,
        catalog_amount_ht=catalog,
    )
