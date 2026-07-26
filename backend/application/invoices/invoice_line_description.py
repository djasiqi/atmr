"""Libellés de ligne facture — source unique pour preview + génération."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from infrastructure.invoices.invoice_description_builder import (
    InvoiceDescriptionBuilder,
)
from models import Booking


def resolve_patient_name_for_invoice(
    client: Any | None, reservations: list[Any]
) -> str:
    """Même logique que GenerateInvoiceUseCase (nom affiché pour tierce / institution)."""
    patient_name = ""
    if client and getattr(client, "is_institution", False) and reservations:
        for _r in reservations:
            if getattr(_r, "customer_name", None):
                patient_name = str(_r.customer_name).strip()
                break
    if not patient_name and client and getattr(client, "user", None):
        u = client.user
        patient_name = (
            f"{getattr(u, 'first_name', '')} {getattr(u, 'last_name', '')}".strip()
        )
    if not patient_name and client:
        cid = getattr(client, "id", None)
        if cid is not None:
            patient_name = f"Client #{cid}"
    if not patient_name:
        patient_name = "Client"
    return patient_name


def booking_source_type_for_preview(booking: Booking) -> str:
    """ride | service | manual — aligné évolution produit."""
    mt = (
        getattr(booking, "mission_type", None) or "patient_transport"
    ) or "patient_transport"
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
    _fee_pct = (
        getattr(bobj, "cancellation_fee_percent", None)
        if bobj and _is_cancelled
        else None
    )
    _fee_tier = (
        getattr(bobj, "cancellation_fee_tier_id", None)
        if bobj and _is_cancelled
        else None
    )

    show_patient = (
        bool(bill_to_client_id or clinic_company_id or billing_party_id)
        and not is_delivery
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
    _is_cancelled_c = (
        str(getattr(reservation, "status", "") or "").upper() == "CANCELED"
    )
    _fee_pct_c = (
        getattr(reservation, "cancellation_fee_percent", None)
        if _is_cancelled_c
        else None
    )
    _fee_tier_c = (
        getattr(reservation, "cancellation_fee_tier_id", None)
        if _is_cancelled_c
        else None
    )
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
    """Libellé A/R : on garde la description du segment aller telle quelle.

    Le tag « [A/R] » est ajouté par le rendu (PDF / aperçu HTML) via le
    line_meta, pour éviter le double libellé « Trajet : aller-retour — ... »
    qui masque la vraie destination.
    """
    base = (primary_segment_description or "").strip()
    if base:
        return base
    if segment_bookings:
        ordered = sorted(
            segment_bookings,
            key=lambda b: (
                b.scheduled_time or datetime.min,
                int(b.id or 0),
            ),
        )
        first = ordered[0]
        pu = (getattr(first, "pickup_location", None) or "").strip()
        do = (getattr(first, "dropoff_location", None) or "").strip()
        if pu and do:
            return f"Trajet {pu} → {do}"
    return "Trajet aller-retour"


def _capitalize_given_names(given: str) -> str:
    given = given.strip()
    if not given:
        return given
    out: list[str] = []
    for part in given.split():
        if "-" in part:
            out.append("-".join(seg.capitalize() for seg in part.split("-")))
        else:
            out.append(part.capitalize())
    return " ".join(out)


def _format_family_name_token(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    if "-" in token:
        return "-".join(seg.upper() for seg in token.split("-"))
    return token.upper()


def _is_family_name_token(token: str) -> bool:
    """Heuristique : token en majuscules (éventuellement hyphené) = nom de famille."""
    t = (token or "").strip().rstrip(".")
    if not t or len(t) < 2:
        return False
    if "-" in t:
        return t.upper() == t
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 2 and t.upper() == t


def format_patient_display_name_nom_prenom(raw: str | None) -> str:
    """Format unifié facture clinique S2 : « NOM Prénom » (nom de famille en tête)."""
    if raw is None:
        return ""
    s = " ".join(str(raw).split())
    if not s:
        return s
    parts = s.split()
    if len(parts) == 1:
        return _format_family_name_token(parts[0])

    if _is_family_name_token(parts[0]):
        nom = _format_family_name_token(parts[0])
        prenom = _capitalize_given_names(" ".join(parts[1:]))
        return f"{nom} {prenom}".strip()

    if _is_family_name_token(parts[-1]):
        nom = _format_family_name_token(parts[-1])
        prenom = _capitalize_given_names(" ".join(parts[:-1]))
        return f"{nom} {prenom}".strip()

    # Convention réservation : dernier token = nom de famille
    nom = _format_family_name_token(parts[-1])
    prenom = _capitalize_given_names(" ".join(parts[:-1]))
    return f"{nom} {prenom}".strip()


def _resolve_institution_patient_name_for_booking(reservation: Booking) -> str:
    """Patient institution bénéficiaire (demande source), ou '' si livraison établissement."""
    fn = getattr(reservation, "_get_institution_passenger_brief", None)
    if callable(fn):
        brief = fn()
        if isinstance(brief, dict) and brief.get("institution_patient_id"):
            first = (brief.get("first_name") or "").strip()
            last = (brief.get("last_name") or "").strip()
            if last or first:
                raw = f"{first} {last}".strip() if first and last else (last or first)
                return format_patient_display_name_nom_prenom(raw)
    return ""


def resolve_s2_clinic_line_patient_name(
    client: Any | None, reservation: Booking
) -> str:
    """Nom « Client : … » pour facture clinique S2 — format unifié « NOM Prénom ».

    Livraison matériel : afficher le client uniquement si un patient institution est
    lié à la demande source ; sinon pas de ligne (livraison pour l'établissement —
    le contact clinique ne doit pas apparaître).
    """
    mission_type = getattr(reservation, "mission_type", None) or "patient_transport"
    if mission_type == "material_delivery":
        return _resolve_institution_patient_name_for_booking(reservation)
    if client and getattr(client, "is_institution", False):
        cn = (getattr(reservation, "customer_name", None) or "").strip()
        inst = (getattr(client, "institution_name", None) or "").strip()
        if not cn:
            return ""
        if inst and cn.lower() == inst.lower():
            return ""
        return format_patient_display_name_nom_prenom(cn)
    if client and getattr(client, "user", None):
        first_name = (client.user.first_name or "").strip()
        last_name = (client.user.last_name or "").strip()
        if last_name and first_name:
            return (
                f"{_format_family_name_token(last_name)} "
                f"{_capitalize_given_names(first_name)}"
            ).strip()
        if last_name:
            return _format_family_name_token(last_name)
        if first_name:
            return _capitalize_given_names(first_name)
        un = getattr(client.user, "username", None)
        if un:
            return str(un)
    cn = getattr(reservation, "customer_name", None)
    if cn and str(cn).strip():
        return format_patient_display_name_nom_prenom(str(cn).strip())
    cid = getattr(reservation, "client_id", None)
    return f"Client #{cid}" if cid is not None else "Client"


def patient_display_name_clinic_monthly(
    client: Any | None, reservation: Booking
) -> str:
    """Même logique que le cache patient dans GenerateClinicMonthlyInvoiceUseCase (aperçu / labels)."""
    return resolve_s2_clinic_line_patient_name(client, reservation)
