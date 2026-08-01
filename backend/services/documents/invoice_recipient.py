"""Résolution du bénéficiaire d'une facture pour le bloc « Facturé à ».

Partagé entre le rendu PDF ReportLab et le constructeur de template HTML, afin que
les deux chemins produisent le même destinataire.

Invariant : le patient d'une facture se déduit uniquement de CETTE facture
(``Invoice.institution_patient_id`` puis ses bookings). Le déduire à l'échelle de
l'institution donnerait la même adresse à tous les résidents.
"""

from __future__ import annotations

from typing import Any


def iter_invoice_bookings(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
):
    """Bookings rattachés aux lignes de la facture.

    ``bookings_by_id`` est fourni par le pipeline PDF pour éviter les N+1 ; sans lui
    on retombe sur une requête ponctuelle par ligne.
    """
    from models import Booking

    for line in getattr(invoice, "lines", None) or []:
        # billed_booking est un backref InstrumentedList, pas un objet unique
        related = getattr(line, "billed_booking", None)
        if isinstance(related, list):
            if related:
                yield related[0]
            continue
        if related is not None:
            yield related
            continue
        reservation_id = getattr(line, "reservation_id", None)
        if not reservation_id:
            continue
        booking = (
            bookings_by_id.get(reservation_id)
            if bookings_by_id is not None
            else Booking.query.get(reservation_id)
        )
        if booking is not None:
            yield booking


def resolve_invoice_institution_patient(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
):
    """Patient institutionnel bénéficiaire de la facture, ou ``None``."""
    from models.institution_patient import InstitutionPatient

    patient_id = getattr(invoice, "institution_patient_id", None)
    if patient_id is None:
        for booking in iter_invoice_bookings(invoice, bookings_by_id=bookings_by_id):
            candidate = getattr(booking, "institution_patient_id", None)
            if candidate is None:
                resolve = getattr(booking, "_resolve_source_transport_request", None)
                request = resolve() if callable(resolve) else None
                candidate = getattr(request, "patient_id", None) if request else None
            if candidate is not None:
                patient_id = candidate
                break
    if patient_id is None:
        return None
    return InstitutionPatient.query.get(int(patient_id))


def institution_patient_billing_address(patient: Any) -> str:
    """Adresse de domicile du patient, qui fait office d'adresse de facturation."""
    if patient is None:
        return ""
    street = (getattr(patient, "address", None) or "").strip()
    postal = (getattr(patient, "postal_code", None) or "").strip()
    city = (getattr(patient, "city", None) or "").strip()
    postal_city = " ".join(part for part in (postal, city) if part)
    return ", ".join(part for part in (street, postal_city) if part)
