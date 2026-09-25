"""E-mail « Transport confirmé / contrat conclu » après commit 7B.5.

N'annule jamais le contrat si l'envoi échoue — notification retryable.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import current_app
from flask_mail import Message

from models.portal_transport_contract_formed import PortalTransportContractFormed

logger = logging.getLogger(__name__)


def build_transport_contract_formed_body(
    *,
    booking: Any,
    contract: PortalTransportContractFormed,
) -> str:
    when = getattr(booking, "scheduled_time", None) or "Non précisée"
    pickup = getattr(booking, "pickup_location", "") or ""
    dropoff = getattr(booking, "dropoff_location", "") or ""
    lines = [
        "Transport confirmé — contrat conclu",
        "",
        "Cet e-mail confirme le contrat de transport déjà formé. "
        "Il ne constitue pas une nouvelle demande de consentement.",
        "",
        f"Référence : #{getattr(booking, 'id', '')}",
        f"Transporteur : {contract.carrier_legal_name}",
        f"Prix contractuel : CHF {contract.carrier_quote}",
        f"Plafond précédemment accepté : CHF {contract.client_ceiling}",
        f"Date et heure : {when}",
        f"Départ : {pickup}",
        f"Destination : {dropoff}",
        "",
        "Conditions d'annulation applicables :",
        str(contract.company_policy_snapshot or "").strip() or "(non fournies)",
        "",
        "Cadre canal LIRIE (plafonds) :",
        str(contract.channel_policy_snapshot or "").strip() or "(non fourni)",
        "",
        "Facturation : par le transporteur après le transport. "
        "LIRIE n'encaisse pas le prix du transport.",
    ]
    return "\n".join(lines)


def notify_portal_transport_contract_formed(
    *,
    booking: Any,
    contract: PortalTransportContractFormed,
    recipient_email: str | None,
) -> bool:
    """Envoie après commit. Retourne True si envoyé. Échec ≠ rollback contrat."""
    recipient = (recipient_email or "").strip() or None
    if not recipient:
        logger.warning(
            "portal contract formed email skipped (no recipient) booking_id=%s",
            getattr(booking, "id", None),
        )
        return False
    try:
        sender = current_app.config.get(
            "MAIL_DEFAULT_SENDER"
        ) or current_app.config.get("MAIL_USERNAME")
        msg = Message(
            subject=(
                f"Transport confirmé — LIRIE #{getattr(booking, 'id', '')} — "
                f"CHF {contract.carrier_quote}"
            ),
            sender=sender,
            recipients=[recipient],
            body=build_transport_contract_formed_body(
                booking=booking, contract=contract
            ),
        )
        from ext import mail

        mail.send(msg)
        return True
    except Exception:
        logger.exception(
            "portal contract formed email failed booking_id=%s",
            getattr(booking, "id", None),
        )
        return False
