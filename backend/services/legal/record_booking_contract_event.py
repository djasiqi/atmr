"""Événement initial de commande PORTAL, dans la transaction du booking.

Le montant figé est l'estimation affichée, pas un prix contractuel.
Sans acceptation CGU/CGV, les clés restent vides : aucune acceptation n'est inventée.

Le débiteur nominal d'une commande PORTAL est le titulaire authentifié du compte.
``billed_to_type`` reste une catégorie opérationnelle et n'est pas cette identité.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func

from ext import db
from models.client import Client
from models.client_booking_contract_event import (
    ACTOR_CLIENT,
    CARRIER_ASSIGNED,
    CARRIER_NOT_ASSIGNED,
    DEBTOR_ACCOUNT_HOLDER,
    DEBTOR_RESOLVED,
    EVENT_BOOKING_CANCELLED,
    EVENT_BOOKING_CREATED,
    EVENT_BOOKING_MODIFIED,
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


def _resolve_portal_account_holder(*, booking: Any, user: User) -> tuple[Client, str]:
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


# Champs visibles ou demandés par le client, et qui décrivent la prestation
# ou son estimation. Le use case de mise à jour n'applique pas le retour,
# l'aller-retour ni le fauteuil : un payload qui ne contient qu'eux ne change
# pas le booking et ne crée pas d'événement.
CLIENT_ORDER_FIELDS = (
    "pickup_location",
    "dropoff_location",
    "scheduled_time",
    "amount",
    "is_round_trip",
    "wheelchair_need",
    "medical_facility",
    "doctor_name",
    "notes_medical",
)


def client_visible_booking_state(booking: Any) -> dict[str, Any]:
    """État client comparable, capturé avant une mutation."""
    state = {field: getattr(booking, field, None) for field in CLIENT_ORDER_FIELDS}
    state["_status"] = _status_text(getattr(booking, "status", None))
    return state


def _same_order_value(field: str, before: Any, after: Any) -> bool:
    if field == "amount":
        try:
            return abs(float(before or 0) - float(after or 0)) < 0.001
        except (TypeError, ValueError):
            return before == after
    if field in {"medical_facility", "doctor_name", "notes_medical"}:
        return str(before or "").strip() == str(after or "").strip()
    if field in {"pickup_location", "dropoff_location"}:
        return str(before or "").strip() == str(after or "").strip()
    if field in {"is_round_trip", "wheelchair_need"}:
        return bool(before) == bool(after)
    return before == after


def changed_client_order_fields(before: dict[str, Any], booking: Any) -> list[str]:
    """Champs de commande réellement différents après la mutation."""
    changed: list[str] = []
    for field in CLIENT_ORDER_FIELDS:
        if not _same_order_value(
            field, before.get(field), getattr(booking, field, None)
        ):
            changed.append(field)
    return changed


def _status_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _portal_account_holder_action(booking: Any, actor_user_id: int) -> bool:
    """Vrai seulement si l'acteur est le titulaire PORTAL de cette réservation."""
    client_id = getattr(booking, "client_id", None)
    client = db.session.get(Client, client_id) if client_id is not None else None
    if client is None or not is_portal_client(client):
        return False
    if int(client.user_id) != int(actor_user_id):
        return False
    booking_user_id = getattr(booking, "user_id", None)
    return booking_user_id is None or int(booking_user_id) == int(actor_user_id)


def _lock_booking_sequence(booking_id: int) -> int:
    """Sérialise les mutations d'une réservation, puis alloue la séquence suivante."""
    from models.booking import Booking

    db.session.query(Booking).filter_by(id=booking_id).with_for_update().one()
    current = (
        db.session.query(func.max(ClientBookingContractEvent.sequence_number))
        .filter(ClientBookingContractEvent.booking_id == booking_id)
        .scalar()
    )
    return int(current or 0) + 1


def _created_terms(
    booking_id: int,
) -> tuple[int | None, int | None, datetime | None]:
    """Conditions et heure de retour figées à la création, sans les réinventer."""
    created = (
        ClientBookingContractEvent.query.filter_by(
            booking_id=booking_id, event_type=EVENT_BOOKING_CREATED
        )
        .order_by(ClientBookingContractEvent.id.asc())
        .first()
    )
    if created is None:
        return None, None, None
    return (
        created.terms_of_service_acceptance_id,
        created.transport_terms_acceptance_id,
        created.return_scheduled_time_snapshot,
    )


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
    maximum_accepted_amount: float | None = None,
    pricing_ceiling_evidence: dict[str, Any] | None = None,
) -> ClientBookingContractEvent:
    """Insère l'événement BOOKING_CREATED. N'altère aucune preuve existante."""
    user = db.session.get(User, user_id)
    if user is None:
        raise PortalBookingEvidenceError("Utilisateur de la commande introuvable.")
    client, debtor_name = _resolve_portal_account_holder(booking=booking, user=user)

    company_id = getattr(booking, "company_id", None)
    terms = _latest_acceptance(user_id, DOCUMENT_TERMS_OF_SERVICE)
    transport = _latest_acceptance(user_id, DOCUMENT_TRANSPORT_TERMS)
    max_amount = maximum_accepted_amount
    if max_amount is None:
        max_amount = getattr(booking, "_portal_maximum_accepted_amount", None)
    evidence_json = None
    if pricing_ceiling_evidence:
        evidence_json = json.dumps(pricing_ceiling_evidence, ensure_ascii=False, sort_keys=True)
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
        maximum_accepted_amount_snapshot=(
            float(max_amount) if max_amount is not None else None
        ),
        pricing_status=PRICING_ESTIMATED,
        amount_is_contractual=False,
        terms_of_service_acceptance_id=terms.id if terms is not None else None,
        transport_terms_acceptance_id=(transport.id if transport is not None else None),
        status_before=None,
        status_after=None,
        cancellation_reason=None,
        # 7B.2 : preuve du calcul MAX(quotes) figée avec le plafond.
        changed_fields=evidence_json,
    )
    db.session.add(event)
    db.session.flush()
    return event


def _append_followup_event(
    *,
    booking: Any,
    actor_user_id: int,
    event_type: str,
    status_before: str | None,
    status_after: str | None,
    changed_fields: list[str] | None,
    cancellation_reason: str | None,
) -> ClientBookingContractEvent:
    user = db.session.get(User, actor_user_id)
    if user is None:
        raise PortalBookingEvidenceError("Utilisateur de la commande introuvable.")
    client, debtor_name = _resolve_portal_account_holder(booking=booking, user=user)
    terms_id, transport_id, return_time = _created_terms(int(booking.id))
    sequence = _lock_booking_sequence(int(booking.id))
    company_id = getattr(booking, "company_id", None)
    event = ClientBookingContractEvent(
        booking_id=int(booking.id),
        sequence_number=sequence,
        event_type=event_type,
        occurred_at=datetime.now(UTC),
        actor_user_id=actor_user_id,
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
        return_scheduled_time_snapshot=return_time,
        wheelchair_need_snapshot=bool(getattr(booking, "wheelchair_need", False)),
        estimated_amount_snapshot=float(getattr(booking, "amount", 0) or 0),
        pricing_status=PRICING_ESTIMATED,
        amount_is_contractual=False,
        terms_of_service_acceptance_id=terms_id,
        transport_terms_acceptance_id=transport_id,
        status_before=status_before,
        status_after=status_after,
        cancellation_reason=cancellation_reason,
        changed_fields=json.dumps(changed_fields) if changed_fields else None,
    )
    db.session.add(event)
    db.session.flush()
    return event


def record_portal_booking_modified_event(
    *,
    booking: Any,
    actor_user_id: int,
    before_state: dict[str, Any],
) -> ClientBookingContractEvent | None:
    """Insère BOOKING_MODIFIED si la commande PORTAL a réellement changé.

    Les conditions référencées restent celles de BOOKING_CREATED. Aucune
    acceptation n'est créée. Un acteur entreprise ou admin ne produit pas
    d'événement client.
    """
    if not _portal_account_holder_action(booking, actor_user_id):
        return None
    changed = changed_client_order_fields(before_state, booking)
    if not changed:
        return None
    status = _status_text(getattr(booking, "status", None))
    return _append_followup_event(
        booking=booking,
        actor_user_id=actor_user_id,
        event_type=EVENT_BOOKING_MODIFIED,
        status_before=_status_text(before_state.get("_status", status)) or status,
        status_after=status,
        changed_fields=changed,
        cancellation_reason=None,
    )


def record_portal_booking_cancelled_event(
    *,
    booking: Any,
    actor_user_id: int,
    status_before: str,
) -> ClientBookingContractEvent | None:
    """Insère BOOKING_CANCELLED dans la transaction d'annulation PORTAL.

    Le motif reste vide : les routes d'annulation n'en collectent pas.
    """
    if not _portal_account_holder_action(booking, actor_user_id):
        return None
    return _append_followup_event(
        booking=booking,
        actor_user_id=actor_user_id,
        event_type=EVENT_BOOKING_CANCELLED,
        status_before=_status_text(status_before),
        status_after=_status_text(getattr(booking, "status", None)) or "canceled",
        changed_fields=["status"],
        cancellation_reason=None,
    )
