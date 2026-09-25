"""Montant HT facturable canonique pour un booking (preview / generate / totaux)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from application.invoices.booking_status import booking_status_is_canceled
from infrastructure.invoices.invoice_calculator import round_to_5_cents

_TWO = Decimal("0.01")
SOURCE_BOOKING_AMOUNT = "booking.amount"
SOURCE_PORTAL_CONTRACTUAL = "portal.contractual_amount"
SOURCE_CANCELLATION_FEE = "cancellation_fee_amount"
SOURCE_CANCELLATION_UNRESOLVED = "cancellation_fee_unresolved"


def _portal_contractual_amount(booking: Any) -> Decimal | None:
    """Montant contractuel PORTAL (DV confirmation ou 7B.5 formation)."""
    try:
        from services.legal.portal_double_validation import (
            booking_uses_conditional_order,
            booking_uses_double_validation,
        )

        bid = getattr(booking, "id", None)
        if bid is None:
            return None

        if booking_uses_conditional_order(booking):
            from models.portal_transport_contract_formed import (
                PortalTransportContractFormed,
            )

            formed = PortalTransportContractFormed.query.filter_by(
                booking_id=int(bid)
            ).one_or_none()
            if formed is None or getattr(formed, "carrier_quote", None) is None:
                return None
            amount = Decimal(str(formed.carrier_quote)).quantize(_TWO)
            if amount <= 0:
                return None
            return amount

        if not booking_uses_double_validation(booking):
            return None
        from models.portal_client_transport_confirmation import (
            PortalClientTransportConfirmation,
        )

        conf = PortalClientTransportConfirmation.query.filter_by(
            booking_id=int(bid)
        ).one_or_none()
        if conf is None or getattr(conf, "contractual_amount", None) is None:
            return None
        amount = Decimal(str(conf.contractual_amount)).quantize(_TWO)
        if amount <= 0:
            return None
        return amount
    except Exception:
        return None


@dataclass(frozen=True)
class BillableAmount:
    amount_ht: Decimal
    source: str
    cancellation_fee_applied: bool
    catalog_amount_ht: Decimal | None
    resolved: bool = True


def calculate_billable_booking_amount(
    booking: Any,
    *,
    billing_settings: Any = None,
    override: dict[str, Any] | None = None,
) -> BillableAmount:
    """Calcule le HT facturable (annulation = frais, livraison = prix fixe settings)."""
    mission_type = getattr(booking, "mission_type", None) or "patient_transport"

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
        return BillableAmount(
            amount_ht=round_to_5_cents(amount),
            source="material_delivery_fixed",
            cancellation_fee_applied=False,
            catalog_amount_ht=None,
            resolved=True,
        )

    catalog = Decimal(str(getattr(booking, "amount", None) or 0)).quantize(_TWO)
    cancellation_fee_applied = False
    resolved = True

    if booking_status_is_canceled(booking):
        fee = getattr(booking, "cancellation_fee_amount", None)
        if fee is not None:
            amount = Decimal(str(fee)).quantize(_TWO)
            cancellation_fee_applied = True
            source = SOURCE_CANCELLATION_FEE
        else:
            amount = Decimal("0.00")
            source = SOURCE_CANCELLATION_UNRESOLVED
            resolved = False
    else:
        contractual = _portal_contractual_amount(booking)
        if contractual is not None:
            amount = contractual
            source = SOURCE_PORTAL_CONTRACTUAL
        else:
            amount = catalog
            source = SOURCE_BOOKING_AMOUNT

    if override and override.get("amount") is not None:
        try:
            amount = Decimal(str(override["amount"])).quantize(_TWO)
            source = "override.amount"
            resolved = True
        except Exception:
            pass

    return BillableAmount(
        amount_ht=round_to_5_cents(amount),
        source=source,
        cancellation_fee_applied=cancellation_fee_applied,
        catalog_amount_ht=catalog,
        resolved=resolved,
    )


UNRESOLVED_CANCELLATION_REASON = "montant d'annulation à déterminer"


def partition_invoiceable_bookings(
    bookings: list,
    *,
    billing_settings: Any = None,
) -> tuple[list, list]:
    """Sépare les segments financièrement émissibles des annulations unresolved."""
    invoiceable: list = []
    unresolved: list = []
    for booking in bookings:
        billed = calculate_billable_booking_amount(
            booking, billing_settings=billing_settings
        )
        if billed.resolved:
            invoiceable.append(booking)
        else:
            unresolved.append(booking)
    return invoiceable, unresolved


def unresolved_cancellation_payload(unresolved: list) -> dict[str, Any]:
    ids: list[int] = []
    for booking in unresolved:
        try:
            ids.append(int(booking.id))
        except (TypeError, ValueError, AttributeError):
            continue
    return {
        "count": len(ids),
        "booking_ids": ids,
        "reason": UNRESOLVED_CANCELLATION_REASON,
        "needs_review": True,
    }


def unresolved_cancellation_warnings(unresolved: list) -> list[str]:
    if not unresolved:
        return []
    ids = ", ".join(
        f"#{int(booking.id)}"
        for booking in unresolved
        if getattr(booking, "id", None) is not None
    )
    return [
        f"{len(unresolved)} annulation(s) avec {UNRESOLVED_CANCELLATION_REASON} "
        f"({ids}) — exclue(s) du total, besoin de revue."
    ]
