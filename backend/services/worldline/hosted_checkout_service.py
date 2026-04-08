"""Création hosted checkout MyCheckout pour une réservation client."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ext import db
from models.enums import BookingStatus, PaymentStatus
from models.payment import Payment
from services.worldline.client_factory import (
    get_worldline_api_client,
    get_worldline_merchant_id,
    worldline_configured,
)
from services.worldline.money import chf_amount_to_cents
from services.worldline.return_url import (
    default_worldline_return_url,
    validate_return_url_override,
)

if TYPE_CHECKING:
    from models.booking import Booking
    from models.client import Client
    from models.user import User

logger = logging.getLogger(__name__)


def create_worldline_hosted_checkout(
    *,
    booking: Booking,
    user: User,
    client: Client,
    return_url_override: str | None = None,
) -> dict[str, Any]:
    """Crée ou met à jour une tentative de paiement Worldline pour un booking.

    Returns:
        dict avec payment_id, hosted_checkout_id, redirect_url (partial ou complète)
    """
    if not worldline_configured():
        raise RuntimeError("Worldline n'est pas configuré sur ce serveur")

    if booking.status != BookingStatus.PENDING:
        raise ValueError("Seules les réservations en attente peuvent être payées en ligne")
    if not booking.client_id or booking.client_id != client.id:
        raise ValueError("Cette réservation n'appartient pas à ce client")

    amount = float(booking.amount or 0)
    if amount <= 0:
        raise ValueError("Montant de réservation invalide pour le paiement")

    cents = chf_amount_to_cents(amount)
    if cents < 50:
        raise ValueError("Montant minimum Worldline (50 centimes) non atteint")

    if return_url_override and str(return_url_override).strip():
        return_url = validate_return_url_override(str(return_url_override))
    else:
        return_url = default_worldline_return_url(booking.id)

    wl_client = get_worldline_api_client()
    merchant_id = get_worldline_merchant_id()

    from worldline.connect.sdk.v1.domain.amount_of_money import AmountOfMoney
    from worldline.connect.sdk.v1.domain.create_hosted_checkout_request import (
        CreateHostedCheckoutRequest,
    )
    from worldline.connect.sdk.v1.domain.hosted_checkout_specific_input import (
        HostedCheckoutSpecificInput,
    )
    from worldline.connect.sdk.v1.domain.order import Order
    from worldline.connect.sdk.v1.domain.order_references import OrderReferences

    payment_row = (
        Payment.query.filter_by(
            booking_id=booking.id,
            payment_provider="worldline",
            client_id=client.id,
        )
        .filter(Payment.status == PaymentStatus.PENDING)
        .order_by(Payment.id.desc())
        .first()
    )

    if payment_row is None:
        payment_row = Payment()
        payment_row.amount = amount
        payment_row.method = "credit_card"
        payment_row.user_id = user.id
        payment_row.client_id = client.id
        payment_row.booking_id = booking.id
        payment_row.status = PaymentStatus.PENDING
        payment_row.payment_provider = "worldline"
        db.session.add(payment_row)
        db.session.flush()
    else:
        if payment_row.worldline_hosted_checkout_id:
            try:
                wl_client.v1().merchant(merchant_id).hostedcheckouts().delete(
                    payment_row.worldline_hosted_checkout_id
                )
            except Exception as e:
                logger.info(
                    "Worldline delete ancien hosted checkout ignoré: %s",
                    e,
                    extra={"hosted_checkout_id": payment_row.worldline_hosted_checkout_id},
                )
            payment_row.worldline_hosted_checkout_id = None
            payment_row.worldline_partial_redirect_url = None

    merchant_ref = f"L{booking.id}"
    if len(merchant_ref) > 30:
        merchant_ref = merchant_ref[:30]

    order = Order()
    money = AmountOfMoney()
    money.currency_code = "CHF"
    money.amount = cents
    order.amount_of_money = money
    refs = OrderReferences()
    refs.merchant_reference = merchant_ref
    order.references = refs

    hci = HostedCheckoutSpecificInput()
    hci.return_url = return_url
    hci.locale = (os.getenv("WORLDLINE_CHECKOUT_LOCALE") or "fr_FR").strip()
    hci.show_result_page = False

    req = CreateHostedCheckoutRequest()
    req.order = order
    req.hosted_checkout_specific_input = hci

    resp = wl_client.v1().merchant(merchant_id).hostedcheckouts().create(req)

    hid = resp.hosted_checkout_id
    partial = resp.partial_redirect_url
    if not hid or not partial:
        raise RuntimeError("Réponse Worldline incomplète (hostedCheckoutId / partialRedirectUrl)")

    payment_row.worldline_hosted_checkout_id = hid
    payment_row.worldline_partial_redirect_url = partial
    db.session.commit()

    logger.info(
        "Worldline hosted checkout créé",
        extra={
            "booking_id": booking.id,
            "payment_id": payment_row.id,
            "hosted_checkout_id": hid,
            "client_id": client.id,
        },
    )

    redirect_url = partial
    if partial.startswith("/"):
        gateway = (
            os.getenv("WORLDLINE_CHECKOUT_GATEWAY_BASE", "").strip().rstrip("/")
            or "https://payment.preprod.pay.worldline-solutions.com"
        )
        redirect_url = f"{gateway}{partial}"

    return {
        "payment_id": payment_row.id,
        "booking_id": booking.id,
        "hosted_checkout_id": hid,
        "redirect_url": redirect_url,
        "merchant_reference": merchant_ref,
    }
