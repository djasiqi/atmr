"""Brouillon de ligne facture (transport / livraison) avant persistance — utilisé par ``generate_invoice``."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from application.invoices.invoice_line_description import build_invoice_line_description
from infrastructure.invoices.invoice_calculator import round_to_5_cents
from infrastructure.invoices.invoice_description_builder import (
    InvoiceDescriptionBuilder,
)
from models import Booking, InvoiceLineType


@dataclass(slots=True)
class RideLineDraft:
    """Montants et libellés calculés pour une réservation — sans création de ligne facture."""

    base_amount: Decimal
    line_vat_rate: Decimal
    description: str
    line_adjustment_note: str | None
    mission_type: str
    line_type: InvoiceLineType
    catalog_ht_patient: Decimal | None
    override_had_amount: bool
    cancellation_fee_applied: bool


class MaterialDeliveryPriceNotConfiguredError(Exception):
    """Prix fixe livraison matériel manquant ou invalide."""

    def __init__(self, message: str, booking_id: int):
        super().__init__(message)
        self.booking_id = booking_id


def _adjustment_note_looks_like_global_remise_legacy(note: Any) -> bool:
    if note is None:
        return False
    s = str(note).strip().lower()
    if not s:
        return False
    return (
        "remise commerciale" in s
        or "remise globale" in s
        or "rabais" in s
        or s.startswith("remise ")
    )


def _override_amount_matches_discounted_catalog(
    catalog_ht: Decimal,
    override_ht: Decimal,
    discount_percent: Decimal,
    *,
    tolerance: Decimal | None = None,
) -> bool:
    if catalog_ht <= 0 or discount_percent <= 0:
        return False
    tol = tolerance if tolerance is not None else Decimal("0.05")
    expected = round_to_5_cents(
        catalog_ht * (Decimal("100") - discount_percent) / Decimal("100")
    )
    return abs(override_ht - expected) <= tol


def compute_ride_line_draft(
    reservation: Booking,
    *,
    two_places: Decimal,
    billing_settings_dto: Any,
    overrides_map: dict[int, dict[str, Any]],
    bookings_by_id: dict[int, Booking],
    gd_pct_early: Decimal | None,
    patient_name: str,
    bill_to_client_id: int | None,
    clinic_company_id: int | None,
    billing_party_id: int | None,
    default_vat_rate: Decimal,
    vat_applicable: bool,
    description_builder: InvoiceDescriptionBuilder,
) -> RideLineDraft:
    """Calcule HT, TVA %, description — aligné sur l'ancienne boucle unique de ``GenerateInvoiceUseCase``."""
    mission_type = getattr(reservation, "mission_type", None) or "patient_transport"
    if mission_type == "material_delivery":
        fixed_price = billing_settings_dto.material_delivery_price_fixed
        if fixed_price is None or fixed_price <= 0:
            raise MaterialDeliveryPriceNotConfiguredError(
                f"Prix fixe livraison non configuré (réservation #{reservation.id}).",
                int(reservation.id),
            )
        base_amount = Decimal(str(fixed_price)).quantize(two_places)
    else:
        base_amount = Decimal(str(reservation.amount or 0)).quantize(two_places)
    catalog_ht_patient = base_amount if mission_type != "material_delivery" else None
    override = overrides_map.get(reservation.id)
    if (
        mission_type != "material_delivery"
        and override
        and "amount" in override
        and override["amount"] is not None
    ):
        with contextlib.suppress(InvalidOperation, ValueError, TypeError):
            base_amount = Decimal(str(override["amount"])).quantize(
                two_places, rounding=ROUND_HALF_UP
            )

    line_vat_rate = Decimal("0")
    if vat_applicable:
        if override and override.get("vat_rate") is not None:
            try:
                override_vat_rate = Decimal(str(override["vat_rate"])).quantize(
                    Decimal("0.01")
                )
                if override_vat_rate > Decimal("0"):
                    line_vat_rate = override_vat_rate
            except (InvalidOperation, ValueError, TypeError):
                line_vat_rate = default_vat_rate
        else:
            line_vat_rate = default_vat_rate

    booking_obj = bookings_by_id.get(reservation.id)
    cancellation_fee_applied = False
    if (
        booking_obj
        and str(getattr(reservation, "status", "") or "").upper() == "CANCELED"
        and getattr(booking_obj, "cancellation_fee_amount", None) is not None
    ):
        base_amount = Decimal(str(booking_obj.cancellation_fee_amount)).quantize(
            two_places
        )
        cancellation_fee_applied = True

    base_amount = round_to_5_cents(base_amount)

    line_adjustment_note = (
        str(override["note"])[:500] if override and override.get("note") else None
    )
    override_had_amount = bool(
        mission_type != "material_delivery"
        and override
        and "amount" in override
        and override.get("amount") is not None
    )
    if (
        gd_pct_early is not None
        and not cancellation_fee_applied
        and mission_type != "material_delivery"
        and catalog_ht_patient is not None
        and catalog_ht_patient > 0
        and _override_amount_matches_discounted_catalog(
            catalog_ht_patient, base_amount, gd_pct_early
        )
        and (
            _adjustment_note_looks_like_global_remise_legacy(line_adjustment_note)
            or (override_had_amount and line_adjustment_note is None)
        )
    ):
        base_amount = round_to_5_cents(catalog_ht_patient)
        line_adjustment_note = None

    is_delivery = mission_type == "material_delivery"
    _is_cancelled = str(getattr(reservation, "status", "") or "").upper() == "CANCELED"
    description = build_invoice_line_description(
        reservation,
        patient_name=patient_name,
        bill_to_client_id=bill_to_client_id,
        clinic_company_id=clinic_company_id,
        billing_party_id=billing_party_id,
        booking_for_cancellation=booking_obj if _is_cancelled else reservation,
        description_builder=description_builder,
    )

    line_type = (
        InvoiceLineType.MATERIAL_DELIVERY if is_delivery else InvoiceLineType.RIDE
    )
    return RideLineDraft(
        base_amount=base_amount,
        line_vat_rate=line_vat_rate,
        description=description,
        line_adjustment_note=line_adjustment_note,
        mission_type=mission_type,
        line_type=line_type,
        catalog_ht_patient=catalog_ht_patient,
        override_had_amount=override_had_amount,
        cancellation_fee_applied=cancellation_fee_applied,
    )
