"""Libellés de ligne facture — source unique pour preview + génération."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from infrastructure.invoices.invoice_description_builder import (
    InvoiceDescriptionBuilder,
)
from models import Booking


def resolve_patient_name_for_invoice(client: Any | None, reservations: list[Any]) -> str:
    """Même logique que GenerateInvoiceUseCase (nom affiché pour tierce / institution)."""
    patient_name = ""
    if client and getattr(client, "is_institution", False) and reservations:
        for _r in reservations:
            if getattr(_r, "customer_name", None):
                patient_name = str(_r.customer_name).strip()
                break
    if not patient_name and client and getattr(client, "user", None):
        u = client.user
        patient_name = f"{getattr(u, 'first_name', '')} {getattr(u, 'last_name', '')}".strip()
    if not patient_name and client:
        cid = getattr(client, "id", None)
        if cid is not None:
            patient_name = f"Client #{cid}"
    if not patient_name:
        patient_name = "Client"
    return patient_name


def booking_source_type_for_preview(booking: Booking) -> str:
    """ride | service | manual — aligné évolution produit."""
    mt = (getattr(booking, "mission_type", None) or "patient_transport") or "patient_transport"
    if mt == "material_delivery":
        return "service"
    return "ride"


def build_invoice_line_description(
    reservation: Booking,
    *,
    patient_name: str,
    bill_to_client_id: int | None,
    clinic_company_id: int | None,
    billing_party_id: int | None,
    booking_for_cancellation: Booking | None = None,
    description_builder: InvoiceDescriptionBuilder | None = None,
) -> str:
    """Aligné sur GenerateInvoiceUseCase (facturation simple / tierce, pas S2 clinique mensuelle)."""
    builder = description_builder or InvoiceDescriptionBuilder()
    mission_type = getattr(reservation, "mission_type", None) or "patient_transport"
    is_delivery = mission_type == "material_delivery"
    delivery_desc = getattr(reservation, "delivery_description", None) or None
    _is_cancelled = str(getattr(reservation, "status", "") or "").upper() == "CANCELED"
    bobj = booking_for_cancellation or reservation
    _fee_pct = getattr(bobj, "cancellation_fee_percent", None) if bobj and _is_cancelled else None
    _fee_tier = getattr(bobj, "cancellation_fee_tier_id", None) if bobj and _is_cancelled else None

    show_patient = (
        bool(bill_to_client_id or clinic_company_id or billing_party_id) and not is_delivery
    )

    return builder.build_description(
        pickup_location=reservation.pickup_location or "",
        dropoff_location=reservation.dropoff_location or "",
        patient_name=patient_name if show_patient else None,
        bill_to_client_id=bill_to_client_id if not is_delivery else None,
        is_material_delivery=is_delivery,
        delivery_description=delivery_desc,
        is_cancelled=_is_cancelled,
        cancellation_fee_percent=_fee_pct,
        cancellation_fee_label=str(_fee_tier) if _fee_tier is not None else None,
    )


def build_invoice_line_description_clinic_monthly(
    reservation: Booking,
    *,
    description_builder: InvoiceDescriptionBuilder | None = None,
) -> str:
    """Aligné sur GenerateClinicMonthlyInvoiceUseCase (lignes S2)."""
    builder = description_builder or InvoiceDescriptionBuilder()
    mission_type = getattr(reservation, "mission_type", None) or "patient_transport"
    is_delivery = mission_type == "material_delivery"
    delivery_desc = getattr(reservation, "delivery_description", None) or None
    _is_cancelled_c = str(getattr(reservation, "status", "") or "").upper() == "CANCELED"
    _fee_pct_c = getattr(reservation, "cancellation_fee_percent", None) if _is_cancelled_c else None
    _fee_tier_c = getattr(reservation, "cancellation_fee_tier_id", None) if _is_cancelled_c else None
    return builder.build_description(
        pickup_location=reservation.pickup_location or "",
        dropoff_location=reservation.dropoff_location or "",
        patient_name=None,
        bill_to_client_id=None,
        is_material_delivery=is_delivery,
        delivery_description=delivery_desc,
        is_cancelled=_is_cancelled_c,
        cancellation_fee_percent=_fee_pct_c,
        cancellation_fee_label=str(_fee_tier_c) if _fee_tier_c is not None else None,
    )


def build_merged_round_trip_invoice_line_description(
    primary_booking: Booking,
    *,
    primary_segment_description: str,
) -> str:
    """Libellé unique sur la facture pour un A/R regroupé (HT cumulé, une ligne).

    Forme compacte type transport sanitaire : pole / site <-> pole / site d'activite.
    """
    return build_merged_round_trip_invoice_line_description_from_segments(
        [primary_booking],
        primary_segment_description=primary_segment_description,
    )


def build_merged_round_trip_invoice_line_description_from_segments(
    segment_bookings: list[Booking],
    *,
    primary_segment_description: str,
) -> str:
    """Libellé A/R pour 1..n segments (ordre chronologique recommandé).

    Utilise les extrêmes du premier et du dernier segment pour le libellé court.
    """
    if not segment_bookings:
        base = (primary_segment_description or "").strip()
        return base or "Trajet aller-retour [A/R]"
    ordered = sorted(
        segment_bookings,
        key=lambda b: (
            b.scheduled_time or datetime.min,
            int(b.id or 0),
        ),
    )
    first = ordered[0]
    last = ordered[-1]
    pu = (getattr(first, "pickup_location", None) or "").strip()
    do = (getattr(last, "dropoff_location", None) or "").strip()
    if pu and do:
        pu_s = pu.split(",")[0].strip()[:200]
        do_s = do.split(",")[0].strip()[:200]
        if pu_s and do_s:
            return f"Trajet aller-retour — {pu_s} ↔ {do_s}"
    base = (primary_segment_description or "").strip()
    if not base:
        return "Trajet aller-retour [A/R]"
    return f"{base} [A/R]"


def patient_display_name_clinic_monthly(client: Any | None, reservation: Booking) -> str:
    """Même logique que le cache patient dans GenerateClinicMonthlyInvoiceUseCase (aperçu / labels)."""
    patient_name = ""
    if client and client.user:
        first_name = (client.user.first_name or "").strip()
        last_name = (client.user.last_name or "").strip()
        if last_name and first_name:
            patient_name = f"{last_name.upper()} {first_name.capitalize()}".strip()
        elif last_name:
            patient_name = last_name.upper()
        elif first_name:
            patient_name = first_name.capitalize()
        else:
            patient_name = (
                client.user.username or f"Client #{reservation.client_id}"
            )
    if not patient_name:
        patient_name = f"Client #{reservation.client_id}"
    return patient_name
