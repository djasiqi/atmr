"""E-mail de confirmation d'une demande PORTAL, après le commit de la commande.

L'échec d'envoi est journalisé. Il ne retire pas le booking ni BOOKING_CREATED.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import current_app
from flask_mail import Message

from ext import db, mail
from models.client_booking_contract_event import (
    EVENT_BOOKING_CREATED,
    ClientBookingContractEvent,
)
from models.client_terms_acceptance import ClientTermsAcceptance
from models.portal_booking_confirmation_email import (
    EMAIL_STATUS_FAILED,
    EMAIL_STATUS_SENT,
    TEMPLATE_PORTAL_BOOKING_CONFIRMATION_V1,
    PortalBookingConfirmationEmail,
)

logger = logging.getLogger(__name__)

_DOCUMENT_LABELS = {
    "terms_of_service": "Conditions générales d'utilisation",
    "transport_terms": "Conditions générales de transport",
}


def _format_when(value: Any) -> str:
    if value is None:
        return "Non précisée"
    return str(value)


def _terms_lines(event: ClientBookingContractEvent | None) -> list[str]:
    if event is None:
        return []
    lines: list[str] = []
    for acceptance_id in (
        event.terms_of_service_acceptance_id,
        event.transport_terms_acceptance_id,
    ):
        if acceptance_id is None:
            continue
        row = db.session.get(ClientTermsAcceptance, acceptance_id)
        if row is None:
            continue
        label = _DOCUMENT_LABELS.get(row.document_type, row.document_type)
        lines.append(f"{label} version {row.terms_version}")
    return lines


def build_confirmation_body(
    *, booking: Any, event: ClientBookingContractEvent | None
) -> str:
    """Texte de confirmation. L'estimation n'y est pas un prix accepté."""
    trip = (
        "Aller-retour" if getattr(booking, "is_round_trip", False) else "Aller simple"
    )
    if event is not None and event.is_round_trip_snapshot:
        trip = "Aller-retour"
    amount = (
        event.estimated_amount_snapshot
        if event is not None
        else getattr(booking, "amount", None)
    )
    debtor = (
        event.debtor_name_snapshot
        if event is not None and event.debtor_name_snapshot
        else "Titulaire du compte"
    )
    pickup = (
        event.pickup_snapshot
        if event is not None
        else getattr(booking, "pickup_location", "")
    )
    dropoff = (
        event.dropoff_snapshot
        if event is not None
        else getattr(booking, "dropoff_location", "")
    )
    when = _format_when(
        event.scheduled_time_snapshot
        if event is not None
        else getattr(booking, "scheduled_time", None)
    )
    status = getattr(getattr(booking, "status", None), "value", None) or getattr(
        booking, "status", "pending"
    )
    lines = [
        "Votre demande de transport a été enregistrée.",
        "Cet e-mail confirme l'enregistrement. Il ne constitue pas la commande.",
        "",
        f"Référence : #{getattr(booking, 'id', '')}",
        f"Date et heure : {when}",
        f"Départ : {pickup}",
        f"Destination : {dropoff}",
        f"Type de trajet : {trip}",
        f"Estimation affichée : CHF {amount} — indicative, non contractuelle.",
        "Le montant final sera facturé par l'entreprise de transport.",
        f"Facturé à : {debtor}",
        f"Statut : {status}",
        "Transporteur : attribué après confirmation.",
    ]
    terms = _terms_lines(event)
    if terms:
        lines.append("")
        lines.append("Conditions applicables :")
        lines.extend(terms)
    return "\n".join(lines)


def notify_portal_booking_confirmed(*, booking: Any, user: Any) -> None:
    """Envoie la confirmation puis trace le résultat. N'annule pas la commande."""
    event = ClientBookingContractEvent.query.filter_by(
        booking_id=int(booking.id),
        event_type=EVENT_BOOKING_CREATED,
    ).one_or_none()
    recipient = str(getattr(user, "email", "") or "").strip() or None
    body = build_confirmation_body(booking=booking, event=event)
    status = EMAIL_STATUS_FAILED
    error: str | None = None
    if not recipient:
        error = "Adresse e-mail absente."
    else:
        try:
            sender = current_app.config.get(
                "MAIL_DEFAULT_SENDER"
            ) or current_app.config.get("MAIL_USERNAME")
            msg = Message(
                subject=(
                    f"Demande de transport enregistrée — LIRIE #{getattr(booking, 'id', '')}"
                ),
                sender=sender,
                recipients=[recipient],
                body=body,
            )
            mail.send(msg)
            status = EMAIL_STATUS_SENT
        except Exception as exc:
            error = str(exc)[:1000]
            logger.exception(
                "Échec e-mail de confirmation PORTAL booking_id=%s",
                getattr(booking, "id", None),
            )
    trace = PortalBookingConfirmationEmail(
        booking_id=int(booking.id),
        contract_event_id=event.id if event is not None else None,
        recipient_email=recipient,
        template_version=TEMPLATE_PORTAL_BOOKING_CONFIRMATION_V1,
        status=status,
        error_message=error,
    )
    db.session.add(trace)
    registry = getattr(db.session, "registry", None)
    real_session = registry() if registry is not None else db.session
    nested = real_session.in_nested_transaction()
    try:
        db.session.flush()
        if not nested:
            db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception(
            "Trace e-mail non enregistrée booking_id=%s ; la commande reste en place",
            getattr(booking, "id", None),
        )
