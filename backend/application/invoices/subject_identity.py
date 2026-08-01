"""Identité de regroupement facturable (sujet) pour un booking.

Règle C1 : un booking institutionnel sans ``institution_patient_id`` ne doit
jamais retomber sur ``client:{carrier}`` (ex. client_id=23 partagé).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from models.enums import BookingCreatedVia

IdentityStatus = Literal["resolved", "needs_review"]


@dataclass(frozen=True)
class ResolvedSubject:
    key: str
    status: IdentityStatus
    subject_type: str
    subject_id: int | None = None
    carrier_client_id: int | None = None


def booking_is_institution_origin(booking: Any) -> bool:
    """True si le booking provient d'une institution (portail / client institution / TR)."""
    created_via = getattr(booking, "created_via", None)
    via_val = (
        created_via.value if hasattr(created_via, "value") else str(created_via or "")
    )
    if via_val == BookingCreatedVia.INSTITUTION_PORTAL.value:
        return True

    client = getattr(booking, "client", None)
    if client is not None:
        if bool(getattr(client, "is_institution", False)):
            return True
        if getattr(client, "linked_institution_id", None) is not None:
            return True

    # TransportRequest source (aller, ou via parent / route_group)
    resolve = getattr(booking, "_resolve_source_transport_request", None)
    if callable(resolve):
        try:
            if resolve() is not None:
                return True
        except Exception:
            pass
    return False


def resolve_subject_identity(booking: Any) -> ResolvedSubject:
    """Résout la clé de sujet facturable pour un booking."""
    carrier = getattr(booking, "client_id", None)
    carrier_id = int(carrier) if carrier is not None else None

    ipid = getattr(booking, "institution_patient_id", None)
    if ipid is not None:
        pid = int(ipid)
        return ResolvedSubject(
            key=f"institution_patient:{pid}",
            status="resolved",
            subject_type="institution_patient",
            subject_id=pid,
            carrier_client_id=carrier_id,
        )

    if booking_is_institution_origin(booking):
        bid = int(getattr(booking, "id", 0) or 0)
        return ResolvedSubject(
            key=f"legacy-institution-booking:{bid}",
            status="needs_review",
            subject_type="legacy_institution_booking",
            subject_id=bid if bid else None,
            carrier_client_id=carrier_id,
        )

    if carrier_id is not None:
        return ResolvedSubject(
            key=f"client:{carrier_id}",
            status="resolved",
            subject_type="client",
            subject_id=carrier_id,
            carrier_client_id=carrier_id,
        )

    bid = int(getattr(booking, "id", 0) or 0)
    return ResolvedSubject(
        key=f"unknown-booking:{bid}",
        status="needs_review",
        subject_type="unknown",
        subject_id=bid if bid else None,
        carrier_client_id=None,
    )
