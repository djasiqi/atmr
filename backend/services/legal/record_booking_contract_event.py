"""Événement initial de commande PORTAL, dans la transaction du booking.

Le montant figé est l'estimation affichée, pas un prix contractuel.
Sans acceptation CGU/CGV, les clés restent vides : aucune acceptation n'est inventée.

Le débiteur nominal d'une commande PORTAL est le titulaire authentifié du compte.
``billed_to_type`` reste une catégorie opérationnelle et n'est pas cette identité.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ext import db
from models.client import Client
from models.client_booking_contract_event import (
    ACTOR_CLIENT,
    CARRIER_ASSIGNED,
    CARRIER_NOT_ASSIGNED,
    DEBTOR_ACCOUNT_HOLDER,
    DEBTOR_RESOLVED,
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
from services.auth.portal_phone_verification import is_portal_client

CLIENT_SUPPLIED_DEBTOR_FIELDS = frozenset(
    {
        "debtor_billing_address",
        "debtor_billing_address_snapshot",
        "debtor_email",
        "debtor_email_snapshot",
        "debtor_name",
        "debtor_name_snapshot",
        "debtor_phone",
        "debtor_phone_snapshot",
        "debtor_resolution",
        "debtor_type",
        "debtor_type_snapshot",
        "debtor_user_id",
    }
)


class PortalBookingEvidenceError(Exception):
    """La preuve de commande n'a pas pu être écrite."""


class ClientSuppliedDebtorError(Exception):
    """Le client a tenté de choisir l'identité du débiteur."""

    code = "client_supplied_debtor_forbidden"


def reject_client_supplied_debtor(payload: dict[str, Any] | None) -> None:
    """Refuse une identité de débiteur fournie par le navigateur."""
    if not payload:
        return
    supplied = CLIENT_SUPPLIED_DEBTOR_FIELDS.intersection(payload)
    if supplied:
        fields = ", ".join(sorted(supplied))
        raise ClientSuppliedDebtorError(
            f"L'identité du débiteur est fixée par le serveur: {fields}."
        )


def _account_holder_name(user: User) -> str:
    first = str(getattr(user, "first_name", "") or "").strip()
    last = str(getattr(user, "last_name", "") or "").strip()
    full = f"{first} {last}".strip()
    if not full:
        raise PortalBookingEvidenceError(
            "Le titulaire du compte n'a pas d'identité nominale."
        )
    return full


def _billing_address(client: Client) -> str | None:
    raw = getattr(client, "billing_address", None)
    if raw is None:
        return None
    address = str(raw).strip()
    return address or None


def _resolve_portal_account_holder(
    *, booking: Any, user: User
) -> tuple[Client, str]:
    """Résout le titulaire PORTAL. N'utilise pas ``billed_to_type``."""
    client_id = getattr(booking, "client_id", None)
    client = db.session.get(Client, client_id) if client_id is not None else None
    if client is None or not is_portal_client(client):
        raise PortalBookingEvidenceError(
            "La résolution du débiteur nominal est réservée au compte PORTAL."
        )
    if int(client.user_id) != int(user.id):
        raise PortalBookingEvidenceError(
            "Le débiteur doit être le titulaire du compte de la réservation."
        )
    booking_user_id = getattr(booking, "user_id", None)
    if booking_user_id is not None and int(booking_user_id) != int(user.id):
        raise PortalBookingEvidenceError(
            "Le débiteur doit être le titulaire du compte de la réservation."
        )
    return client, _account_holder_name(user)


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
    client, debtor_name = _resolve_portal_account_holder(booking=booking, user=user)

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
        debtor_resolution=DEBTOR_RESOLVED,
        debtor_type_snapshot=DEBTOR_ACCOUNT_HOLDER,
        debtor_user_id=int(user.id),
        debtor_name_snapshot=debtor_name,
        debtor_email_snapshot=getattr(user, "email", None),
        debtor_phone_snapshot=getattr(user, "phone", None),
        debtor_billing_address_snapshot=_billing_address(client),
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
