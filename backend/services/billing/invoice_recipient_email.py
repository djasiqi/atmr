"""Résolution e-mail destinataire pour envoi de facture (Direct patient / PORTAL)."""

from __future__ import annotations

from typing import Any


def _clean_email(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or "@" not in value:
        return None
    return value


def _load_client_with_user(client_id: int):
    from sqlalchemy.orm import joinedload

    from models.client import Client

    return (
        Client.query.options(joinedload(Client.user))
        .filter_by(id=int(client_id))
        .first()
    )


def _email_from_client(client: Any) -> str | None:
    if client is None:
        return None
    user = getattr(client, "user", None)
    found = _clean_email(getattr(user, "email", None) if user is not None else None)
    if found:
        return found
    return _clean_email(getattr(client, "contact_email", None))


def resolve_invoice_recipient_email(invoice: Any) -> str | None:
    """Chaîne de résolution e-mail pour ``SendEmailModal`` / envoi.

    Ordre :
    1. e-mail déjà figé sur la facture (meta / champ dédié si présent)
    2. compte client PORTAL / portefeuille (``user.email`` puis ``contact_email``)
    3. ``BillingParty.contact_email``
    4. snapshot débiteur ``BOOKING_CREATED.debtor_email_snapshot`` (1re course liée)
    """
    if invoice is None:
        return None

    meta = getattr(invoice, "meta", None)
    if isinstance(meta, dict):
        for key in ("recipient_email", "last_recipient_email", "debtor_email"):
            found = _clean_email(meta.get(key))
            if found:
                return found

    # Client déjà hydraté (tests / réponse API)
    found = _email_from_client(getattr(invoice, "client", None))
    if found:
        return found

    # Rechargement avec user (stub list_view sans relation user)
    client_id = getattr(invoice, "client_id", None)
    if client_id is None:
        rel = getattr(invoice, "client", None)
        if rel is not None and getattr(rel, "id", None):
            client_id = rel.id
    if client_id is not None:
        try:
            found = _email_from_client(_load_client_with_user(int(client_id)))
            if found:
                return found
        except Exception:
            pass

    bp = getattr(invoice, "billing_party", None)
    if bp is None and getattr(invoice, "billing_party_id", None):
        try:
            from models import BillingParty

            bp = BillingParty.query.get(int(invoice.billing_party_id))
        except Exception:
            bp = None
    if bp is not None:
        found = _clean_email(getattr(bp, "contact_email", None))
        if found:
            return found

    # Snapshot contractuel via première ligne course
    try:
        from models.client_booking_contract_event import (
            EVENT_BOOKING_CREATED,
            ClientBookingContractEvent,
        )

        lines = getattr(invoice, "lines", None) or []
        for line in lines:
            rid = getattr(line, "reservation_id", None)
            if rid is None:
                continue
            event = ClientBookingContractEvent.query.filter_by(
                booking_id=int(rid),
                event_type=EVENT_BOOKING_CREATED,
            ).one_or_none()
            if event is None:
                continue
            found = _clean_email(getattr(event, "debtor_email_snapshot", None))
            if found:
                return found
    except Exception:
        pass

    return None
