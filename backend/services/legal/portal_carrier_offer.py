"""Création d'une offre transporteur PORTAL (CARRIER_OFFERED)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ext import db
from models import Company
from models.portal_carrier_offer import (
    OFFER_STATUS_OFFERED,
    OFFER_STATUS_STALE,
    PortalCarrierOffer,
)
from services.billing.portal_booking_debtor import resolve_portal_booking_debtor_user_id
from services.billing.portal_payment_hold import (
    ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
    resolve_portal_payment_hold,
)
from services.legal.portal_cancellation_policy import get_current_cancellation_policy
from services.legal.portal_double_validation import (
    ERROR_PORTAL_NO_CANCELLATION_POLICY,
    ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT,
    ERROR_PORTAL_OFFER_CONFLICT,
    booking_uses_double_validation,
)
from shared.time_utils import now_utc


def _money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _offer_content_hash(
    *,
    booking_id: int,
    company_id: int,
    offered_amount: Decimal,
    currency: str,
    policy_hash: str,
) -> str:
    payload = {
        "booking_id": booking_id,
        "company_id": company_id,
        "offered_amount": str(offered_amount),
        "currency": currency,
        "cancellation_policy_hash": policy_hash,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_maximum_accepted_for_booking(booking: Any) -> Decimal | None:
    """Plafond figé sur BOOKING_CREATED — jamais exposé au transporteur."""
    from models.client_booking_contract_event import (
        EVENT_BOOKING_CREATED,
        ClientBookingContractEvent,
    )

    event = ClientBookingContractEvent.query.filter_by(
        booking_id=int(booking.id),
        event_type=EVENT_BOOKING_CREATED,
    ).one_or_none()
    if event is None:
        return None
    raw = getattr(event, "maximum_accepted_amount_snapshot", None)
    if raw is None:
        return None
    return _money(raw)


@dataclass(frozen=True, slots=True)
class CreateCarrierOfferResult:
    ok: bool
    offer: PortalCarrierOffer | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def create_portal_carrier_offer(
    *,
    booking: Any,
    company_id: int,
    offered_amount: Any,
    actor_user_id: int | None = None,
) -> CreateCarrierOfferResult:
    """Crée une offre ; n'assigne PAS le booking ; ne forme PAS le contrat."""
    if not booking_uses_double_validation(booking):
        return CreateCarrierOfferResult(
            ok=False,
            error={"error": "portal_double_validation_inactive"},
            status_code=400,
        )

    company = Company.query.get(int(company_id))
    if not company or not company.is_approved:
        return CreateCarrierOfferResult(
            ok=False,
            error={"error": "Entreprise non approuvée"},
            status_code=403,
        )

    status_str = getattr(getattr(booking, "status", None), "value", None) or str(
        getattr(booking, "status", "") or ""
    )
    if str(status_str).upper() != "PENDING":
        return CreateCarrierOfferResult(
            ok=False,
            error={"error": "Reservation not found or cannot be accepted"},
            status_code=400,
        )

    if getattr(booking, "company_id", None) is not None:
        return CreateCarrierOfferResult(
            ok=False,
            error={
                "error": ERROR_PORTAL_OFFER_CONFLICT,
                "message": ("Cette course a déjà été confirmée avec un transporteur."),
            },
            status_code=409,
        )

    # Verrou booking pour course à une offre active.
    from models.booking import Booking

    locked = (
        db.session.query(Booking).filter_by(id=int(booking.id)).with_for_update().one()
    )

    existing = (
        PortalCarrierOffer.query.filter_by(
            booking_id=int(locked.id), status=OFFER_STATUS_OFFERED
        )
        .with_for_update()
        .first()
    )
    if existing is not None:
        # Idempotence : même transporteur re-clique « Accepter » → succès, offre existante.
        if int(existing.company_id) == int(company_id):
            return CreateCarrierOfferResult(ok=True, offer=existing)
        return CreateCarrierOfferResult(
            ok=False,
            error={
                "error": ERROR_PORTAL_OFFER_CONFLICT,
                "message": (
                    "Un autre transporteur a déjà proposé cette course. "
                    "Une seule proposition active est autorisée."
                ),
            },
            status_code=409,
        )

    debtor_user_id = resolve_portal_booking_debtor_user_id(locked)
    if debtor_user_id is not None:
        hold = resolve_portal_payment_hold(int(debtor_user_id), int(company_id))
        if hold.is_hold:
            return CreateCarrierOfferResult(
                ok=False,
                error={
                    "error": ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
                    "message": (
                        "Ce client a une facture échue auprès de votre "
                        "entreprise. L'offre est refusée."
                    ),
                },
                status_code=409,
            )

    maximum = get_maximum_accepted_for_booking(locked)
    if maximum is None:
        return CreateCarrierOfferResult(
            ok=False,
            error={"error": "portal_maximum_missing"},
            status_code=409,
        )

    try:
        amount = _money(offered_amount)
    except Exception:
        return CreateCarrierOfferResult(
            ok=False,
            error={"error": "invalid_offer_amount"},
            status_code=400,
        )
    if amount <= Decimal("0.00"):
        return CreateCarrierOfferResult(
            ok=False,
            error={"error": "invalid_offer_amount"},
            status_code=400,
        )

    # Gate strict — ne jamais révéler le plafond au transporteur.
    if amount > maximum:
        return CreateCarrierOfferResult(
            ok=False,
            error={
                "error": ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT,
                "message": (
                    "Le prix proposé dépasse le plafond accepté par le client."
                ),
            },
            status_code=409,
        )

    policy = get_current_cancellation_policy(int(company_id))
    if policy is None:
        return CreateCarrierOfferResult(
            ok=False,
            error={
                "error": ERROR_PORTAL_NO_CANCELLATION_POLICY,
                "message": ("Publiez d'abord vos conditions d'annulation PORTAL."),
            },
            status_code=409,
        )

    currency = "CHF"
    company_name = str(
        getattr(company, "legal_name", None)
        or getattr(company, "name", None)
        or f"Entreprise #{company_id}"
    )
    content_hash = _offer_content_hash(
        booking_id=int(locked.id),
        company_id=int(company_id),
        offered_amount=amount,
        currency=currency,
        policy_hash=str(policy.content_hash),
    )

    offer = PortalCarrierOffer(
        booking_id=int(locked.id),
        company_id=int(company_id),
        company_name_snapshot=company_name,
        offered_amount=amount,
        currency=currency,
        cancellation_policy_id=int(policy.id),
        cancellation_policy_version=str(policy.version),
        cancellation_policy_snapshot=str(policy.body_text),
        cancellation_policy_hash=str(policy.content_hash),
        offer_content_hash=content_hash,
        status=OFFER_STATUS_OFFERED,
        actor_user_id=int(actor_user_id) if actor_user_id else None,
        offered_at=now_utc(),
    )
    db.session.add(offer)
    db.session.flush()
    return CreateCarrierOfferResult(ok=True, offer=offer)


def mark_active_offers_stale(booking_id: int) -> int:
    """Rend STALE toute offre active (modification matérielle, etc.)."""
    rows = PortalCarrierOffer.query.filter_by(
        booking_id=int(booking_id), status=OFFER_STATUS_OFFERED
    ).all()
    for row in rows:
        row.status = OFFER_STATUS_STALE
    return len(rows)
