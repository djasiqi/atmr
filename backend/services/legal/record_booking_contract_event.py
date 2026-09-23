"""Événement initial de commande PORTAL, dans la transaction du booking.

Le montant figé est l'estimation affichée, pas un prix contractuel.
Sans acceptation CGU/CGV, les clés restent vides : aucune acceptation n'est inventée.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ext import db
from models.client_booking_contract_event import (
    ACTOR_CLIENT,
    CARRIER_ASSIGNED,
    CARRIER_NOT_ASSIGNED,
    DEBTOR_PARTIAL,
    EVENT_BOOKING_CREATED,
    PRICING_ESTIMATED,
    ClientBookingContractEvent,
)
from models.client_terms_acceptance import (
    DOCUMENT_TERMS_OF_SERVICE,
    DOCUMENT_TRANSPORT_TERMS,
    ClientTermsAcceptance,
)
from models.user import User


class PortalBookingEvidenceError(Exception):
    """La preuve de commande n'a pas pu être écrite."""


def _latest_acceptance(
    user_id: int, document_type: str
) -> ClientTermsAcceptance | None:
    return (
        ClientTermsAcceptance.query.filter_by(
            user_id=user_id, document_type=document_type
        )
        .order_by(
            ClientTermsAcceptance.accepted_at.desc(),
            ClientTermsAcceptance.id.desc(),
        )
        .first()
    )


def record_portal_booking_created_event(
    *,
    booking: Any,
    user_id: int,
    return_scheduled_time: datetime | None = None,
) -> ClientBookingContractEvent:
    """Insère l'événement BOOKING_CREATED. N'altère aucune preuve existante."""
    user = db.session.get(User, user_id)
    if user is None:
        raise PortalBookingEvidenceError("Utilisateur de la commande introuvable.")

    company_id = getattr(booking, "company_id", None)
    terms = _latest_acceptance(user_id, DOCUMENT_TERMS_OF_SERVICE)
    transport = _latest_acceptance(user_id, DOCUMENT_TRANSPORT_TERMS)
    event = ClientBookingContractEvent(
        booking_id=int(booking.id),
        sequence_number=1,
        event_type=EVENT_BOOKING_CREATED,
        occurred_at=datetime.now(UTC),
        actor_user_id=user_id,
        actor_type=ACTOR_CLIENT,
        customer_name_snapshot=str(getattr(booking, "customer_name", "") or ""),
        email_snapshot=getattr(user, "email", None),
        phone_snapshot=getattr(user, "phone", None),
        passenger_name_snapshot=None,
        billed_to_type_snapshot=str(
            getattr(booking, "billed_to_type", None) or "patient"
        ),
        debtor_resolution=DEBTOR_PARTIAL,
        carrier_status=(CARRIER_ASSIGNED if company_id else CARRIER_NOT_ASSIGNED),
        company_id_snapshot=int(company_id) if company_id else None,
        pickup_snapshot=str(getattr(booking, "pickup_location", "") or ""),
        dropoff_snapshot=str(getattr(booking, "dropoff_location", "") or ""),
        scheduled_time_snapshot=getattr(booking, "scheduled_time", None),
        is_round_trip_snapshot=bool(getattr(booking, "is_round_trip", False)),
        return_scheduled_time_snapshot=return_scheduled_time,
        wheelchair_need_snapshot=bool(getattr(booking, "wheelchair_need", False)),
        estimated_amount_snapshot=float(getattr(booking, "amount", 0) or 0),
        pricing_status=PRICING_ESTIMATED,
        amount_is_contractual=False,
        terms_of_service_acceptance_id=terms.id if terms is not None else None,
        transport_terms_acceptance_id=(transport.id if transport is not None else None),
    )
    db.session.add(event)
    db.session.flush()
    return event
