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


def invoice_residence_label(
    *,
    patient: Any | None = None,
    client: Any | None = None,
) -> str:
    """Établissement de résidence (EMS, foyer) à coller sous le nom « Facturé à »."""
    if patient is not None:
        name = (getattr(patient, "residence_name", None) or "").strip()
        if name:
            return name
    if client is not None:
        return (getattr(client, "residence_facility", None) or "").strip()
    return ""


def append_residence_to_billed_to_name(
    name: str,
    residence: str,
    *,
    separator: str,
) -> str:
    """Ajoute la résidence sous le nom, sans doublon."""
    base = (name or "").strip()
    label = (residence or "").strip()
    if not label:
        return base
    if label.casefold() in base.casefold():
        return base
    return f"{base}{separator}{label}"


def live_patient_payer_identity(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Nom et adresse courants du payeur PATIENT (relecture live, pas snapshot BP).

    Un BillingParty PATIENT est un pointeur vers le bénéficiaire : si le client
    ou le patient institutionnel a été modifié après création de la facture, le
    PDF doit reprendre ces valeurs actuelles — jamais un ``display_name`` /
    ``billing_address`` périmé.
    """
    patient = resolve_invoice_institution_patient(
        invoice, bookings_by_id=bookings_by_id
    )
    if patient is not None:
        first = (getattr(patient, "first_name", None) or "").strip()
        last = (getattr(patient, "last_name", None) or "").strip()
        name = f"{first} {last}".strip() or None
        addr = institution_patient_billing_address(patient) or None
        return name, addr or None

    client = getattr(invoice, "client", None)
    if client is None:
        return None, None

    user = getattr(client, "user", None)
    name = None
    if user is not None:
        first = (getattr(user, "first_name", None) or "").strip()
        last = (getattr(user, "last_name", None) or "").strip()
        full = f"{first} {last}".strip()
        username = (getattr(user, "username", None) or "").strip()
        name = full or username or None
    if not name:
        first = (getattr(client, "first_name", None) or "").strip()
        last = (getattr(client, "last_name", None) or "").strip()
        name = f"{first} {last}".strip() or None
    if not name and bool(getattr(client, "is_institution", False)):
        name = (getattr(client, "institution_name", None) or "").strip() or None

    street = (getattr(client, "domicile_address", None) or "").strip()
    postal = (getattr(client, "domicile_zip", None) or "").strip()
    city = (getattr(client, "domicile_city", None) or "").strip()
    parts: list[str] = []
    if street:
        parts.append(street)
    postal_city = " ".join(part for part in (postal, city) if part)
    if postal_city:
        parts.append(postal_city)
    addr = "\n".join(parts) if parts else None
    if not addr:
        try:
            addr = (
                getattr(client, "billing_address_secure", None) or ""
            ).strip() or None
        except Exception:
            addr = (getattr(client, "billing_address", None) or "").strip() or None
    if not addr and user is not None:
        addr = (getattr(user, "address", None) or "").strip() or None
    return name, addr


def format_billing_party_recipient_name(
    bp_display_name: str,
    contact_name: str | None = None,
    *,
    separator: str = "\n",
) -> str:
    """Nom « Facturé à » : débiteur/organisme, puis contact facturation.

    Le type du tiers payeur (ex. ``curatorship``) ne qualifie jamais le contact
    comme représentant légal. On affiche uniquement :

    ``Hospice général``
    ``À l'att. de Mme Amandine HAUSER``
    """
    name = (bp_display_name or "Payeur").strip() or "Payeur"
    contact = (contact_name or "").strip()
    if not contact:
        return name
    if contact.casefold() == name.casefold():
        return name
    return f"{name}{separator}À l'att. de {contact}"
