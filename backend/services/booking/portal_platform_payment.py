"""Encaissement plateforme pour une réservation client.

Le compte privé ``ClientType.PORTAL`` n'est pas encaissé sur LIRIE : la course
est transmise, puis l'entreprise de transport facture le client selon son
propre processus. Saferpay reste disponible pour les autres comptes client
(notamment ``TRANSPORT``) et pour le parcours invité, qui n'utilise pas ce
module.
"""

from __future__ import annotations

from services.auth.portal_phone_verification import is_portal_client

PORTAL_PLATFORM_PAYMENT_MESSAGE = (
    "Le paiement en ligne n'est pas disponible pour un compte client privé. "
    "L'entreprise de transport facture la course après la prestation."
)


class PortalPlatformPaymentForbidden(Exception):
    """Refus d'encaissement Saferpay pour un client privé PORTAL."""

    code = "portal_platform_payment_forbidden"

    def __init__(self, message: str = PORTAL_PLATFORM_PAYMENT_MESSAGE) -> None:
        self.message = message
        super().__init__(message)


def portal_skips_platform_checkout(client: object | None) -> bool:
    """True si ce client ne doit pas être envoyé vers Saferpay."""
    return is_portal_client(client)


def platform_checkout_forbidden_for(
    client: object | None,
    booking: object | None = None,
) -> bool:
    """True si le client du compte ou le client de la réservation est PORTAL."""
    if portal_skips_platform_checkout(client):
        return True
    if booking is None:
        return False
    return portal_skips_platform_checkout(getattr(booking, "client", None))


def should_hold_client_booking_for_platform_payment(
    client: object | None,
    billed_to_type: str | None,
) -> bool:
    """True si la création doit passer la réservation en attente de paiement.

    ``billed_to_type = patient`` désigne le débiteur, pas un ordre de payer
    sur la plateforme. Pour un compte PORTAL, ce défaut ne déclenche pas
    ``AWAITING_CLIENT_PAYMENT``.
    """
    billed = str(billed_to_type or "patient").strip().lower()
    return billed == "patient" and not portal_skips_platform_checkout(client)
