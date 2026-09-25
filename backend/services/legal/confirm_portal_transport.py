"""Second clic client — CLIENT_TRANSPORT_CONFIRMED (formation du contrat)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ext import db
from models.portal_carrier_offer import (
    OFFER_STATUS_CONFIRMED,
    OFFER_STATUS_OFFERED,
    PortalCarrierOffer,
)
from models.portal_client_transport_confirmation import (
    PortalClientTransportConfirmation,
)
from services.billing.portal_booking_debtor import resolve_portal_booking_debtor_user_id
from services.billing.portal_payment_hold import (
    ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
    resolve_portal_payment_hold,
)
from services.legal.portal_carrier_offer import get_maximum_accepted_for_booking
from services.legal.portal_double_validation import (
    ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT,
    ERROR_PORTAL_OFFER_STALE,
    booking_uses_double_validation,
)
from shared.time_utils import now_utc


def _money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _set_status(obj: Any, status_str: str) -> None:
    current = getattr(obj, "status", None)
    enum_cls = getattr(current, "__class__", None)
    candidate_name = status_str.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate_name):
        obj.status = getattr(enum_cls, candidate_name)
        return
    obj.status = status_str


@dataclass(frozen=True, slots=True)
class ConfirmTransportResult:
    ok: bool
    confirmation: PortalClientTransportConfirmation | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def confirm_portal_transport(
    *,
    booking: Any,
    user_id: int,
    carrier_offer_id: int,
    expected_offer_hash: str | None = None,
) -> ConfirmTransportResult:
    """Forme le contrat : assigne company_id seulement après ce clic."""
    if not booking_uses_double_validation(booking):
        return ConfirmTransportResult(
            ok=False,
            error={"error": "portal_double_validation_inactive"},
            status_code=400,
        )

    from models.booking import Booking

    locked = (
        db.session.query(Booking).filter_by(id=int(booking.id)).with_for_update().one()
    )

    existing_conf = PortalClientTransportConfirmation.query.filter_by(
        booking_id=int(locked.id)
    ).one_or_none()
    if existing_conf is not None:
        # Idempotence : même offre déjà confirmée.
        if int(existing_conf.carrier_offer_id) == int(carrier_offer_id):
            return ConfirmTransportResult(ok=True, confirmation=existing_conf)
        return ConfirmTransportResult(
            ok=False,
            error={"error": "portal_already_confirmed"},
            status_code=409,
        )

    offer = (
        PortalCarrierOffer.query.filter_by(id=int(carrier_offer_id))
        .with_for_update()
        .one_or_none()
    )
    if offer is None or int(offer.booking_id) != int(locked.id):
        return ConfirmTransportResult(
            ok=False,
            error={"error": ERROR_PORTAL_OFFER_STALE},
            status_code=409,
        )
    if offer.status != OFFER_STATUS_OFFERED:
        return ConfirmTransportResult(
            ok=False,
            error={"error": ERROR_PORTAL_OFFER_STALE},
            status_code=409,
        )
    if expected_offer_hash and str(offer.offer_content_hash) != str(
        expected_offer_hash
    ):
        return ConfirmTransportResult(
            ok=False,
            error={"error": ERROR_PORTAL_OFFER_STALE},
            status_code=409,
        )

    status_str = getattr(getattr(locked, "status", None), "value", None) or str(
        getattr(locked, "status", "") or ""
    )
    if str(status_str).upper() in ("CANCELED", "CANCELLED", "COMPLETED"):
        return ConfirmTransportResult(
            ok=False,
            error={"error": ERROR_PORTAL_OFFER_STALE},
            status_code=409,
        )

    # Copie exacte depuis l'offre présentée — jamais depuis booking.amount / plafond.
    offered = _money(offer.offered_amount)
    maximum = get_maximum_accepted_for_booking(locked)
    if maximum is None or offered > maximum:
        return ConfirmTransportResult(
            ok=False,
            error={"error": ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT},
            status_code=409,
        )

    debtor_user_id = resolve_portal_booking_debtor_user_id(locked)
    if debtor_user_id is not None:
        hold = resolve_portal_payment_hold(int(debtor_user_id), int(offer.company_id))
        if hold.is_hold:
            return ConfirmTransportResult(
                ok=False,
                error={
                    "error": ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
                    "message": (
                        "Une facture échue empêche la confirmation "
                        "avec cette entreprise."
                    ),
                },
                status_code=409,
            )

    if debtor_user_id is not None and int(debtor_user_id) != int(user_id):
        return ConfirmTransportResult(
            ok=False,
            error={"error": "forbidden"},
            status_code=403,
        )

    offer_hash = str(offer.offer_content_hash)
    confirmation = PortalClientTransportConfirmation(
        booking_id=int(locked.id),
        carrier_offer_id=int(offer.id),
        company_id=int(offer.company_id),
        company_name_snapshot=str(offer.company_name_snapshot),
        contractual_amount=offer.offered_amount,
        currency=str(offer.currency or "CHF"),
        cancellation_policy_version=str(offer.cancellation_policy_version),
        cancellation_policy_snapshot=str(offer.cancellation_policy_snapshot),
        cancellation_policy_hash=str(offer.cancellation_policy_hash),
        carrier_offer_hash=offer_hash,
        confirmed_by_user_id=int(user_id),
        confirmed_at=now_utc(),
    )
    db.session.add(confirmation)

    offer.status = OFFER_STATUS_CONFIRMED
    locked.company_id = int(offer.company_id)
    locked.amount = float(offered)
    _set_status(locked, "accepted")
    db.session.flush()

    # Destinataire facturation PATIENT (clients PORTAL : company_id client = NULL).
    from services.billing.billing_party_linker import (
        ensure_patient_destination_billing_party,
    )

    ensure_patient_destination_billing_party(locked)

    # Invariants post-écriture
    assert str(confirmation.carrier_offer_hash) == offer_hash
    assert _money(confirmation.contractual_amount) == offered
    return ConfirmTransportResult(ok=True, confirmation=confirmation)
