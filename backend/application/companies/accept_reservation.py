from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _status_value(status: Any) -> str:
    if status is None:
        return ""
    v = getattr(status, "value", None)
    if isinstance(v, str):
        return v
    return str(status)


def _set_status(obj: Any, status_str: str) -> None:
    current = getattr(obj, "status", None)
    enum_cls = getattr(current, "__class__", None)
    candidate_name = status_str.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate_name):
        obj.status = getattr(enum_cls, candidate_name)
        return
    obj.status = status_str


class _BookingLike(Protocol):
    id: int | None
    status: Any
    company_id: Any
    client_id: Any


@dataclass(frozen=True, slots=True)
class AcceptReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False
    # True si PORTAL v2 : offre créée, pas d'assignation définitive.
    portal_offer_pending_client: bool = False
    portal_offer_id: int | None = None
    # True si conditional_order : rejeu idempotent du même contrat.
    idempotent_replay: bool = False


class AcceptReservationUseCase:
    """Accepter une réservation (PENDING -> ACCEPTED).

    Pour PORTAL double_validation_v2 : crée une offre (CARRIER_OFFERED) sans
    assigner ni former le contrat. Les autres flux gardent l'assignation.
    """

    def execute(
        self,
        booking: _BookingLike,
        *,
        company_id: int,
        offered_amount: Any | None = None,
        actor_user_id: int | None = None,
    ) -> AcceptReservationResult:
        from models import Company
        from models.client import Client
        from services.auth.portal_phone_verification import is_portal_client
        from services.billing.portal_booking_debtor import (
            resolve_portal_booking_debtor_user_id,
        )
        from services.billing.portal_payment_hold import (
            ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
            resolve_portal_payment_hold,
        )
        from services.legal.portal_double_validation import (
            booking_uses_conditional_order,
            booking_uses_double_validation,
        )

        company = Company.query.get(company_id)
        if not company or not company.is_approved:
            return AcceptReservationResult(
                ok=False,
                error={"error": "Entreprise non approuvée"},
                status_code=403,
            )

        client = None
        client_id = getattr(booking, "client_id", None)
        if client_id is not None:
            client = Client.query.get(int(client_id))

        # --- PORTAL 7B.5 : formation atomique (avant le gate PENDING) ---
        # form_portal_transport_contract_on_accept possède l'idempotence
        # (même company) et transport_already_assigned (autre), y compris
        # si le booking est déjà ACCEPTED.
        if (
            client is not None
            and is_portal_client(client)
            and booking_uses_conditional_order(booking)
        ):
            from services.legal.form_portal_transport_contract import (
                form_portal_transport_contract_on_accept,
            )
            from services.pricing.portal_carrier_ceiling import (
                estimate_portal_carrier_offer_amount,
            )

            amount = offered_amount
            if amount is None:
                amount = estimate_portal_carrier_offer_amount(booking, int(company_id))
            result = form_portal_transport_contract_on_accept(
                booking=booking,
                company_id=int(company_id),
                offered_amount=amount,
                actor_user_id=actor_user_id,
            )
            if not result.ok:
                return AcceptReservationResult(
                    ok=False,
                    error=result.error,
                    status_code=result.status_code,
                )
            return AcceptReservationResult(
                ok=True,
                should_trigger_dispatch=False,
                idempotent_replay=bool(result.idempotent_replay),
            )

        status_str = _status_value(getattr(booking, "status", None))
        if status_str.upper() != "PENDING":
            return AcceptReservationResult(
                ok=False,
                error={"error": "Reservation not found or cannot be accepted"},
                status_code=400,
            )

        # --- PORTAL v2 : offre en attente de confirmation client ---
        if (
            client is not None
            and is_portal_client(client)
            and booking_uses_double_validation(booking)
        ):
            from services.legal.portal_carrier_offer import create_portal_carrier_offer
            from services.pricing.portal_carrier_ceiling import (
                estimate_portal_carrier_offer_amount,
            )

            amount = offered_amount
            if amount is None:
                # Jamais booking.amount (estimation client) : grille du transporteur.
                amount = estimate_portal_carrier_offer_amount(booking, int(company_id))
            if amount is None:
                return AcceptReservationResult(
                    ok=False,
                    error={
                        "error": "portal_carrier_quote_unavailable",
                        "message": (
                            "Impossible de calculer votre tarif pour cette course. "
                            "Vérifiez votre grille tarifaire active, ou indiquez "
                            "offered_amount."
                        ),
                    },
                    status_code=409,
                )
            result = create_portal_carrier_offer(
                booking=booking,
                company_id=int(company_id),
                offered_amount=amount,
                actor_user_id=actor_user_id,
            )
            if not result.ok:
                return AcceptReservationResult(
                    ok=False,
                    error=result.error,
                    status_code=result.status_code,
                )
            # PAS d'assignation company_id, PAS de statut accepted.
            return AcceptReservationResult(
                ok=True,
                should_trigger_dispatch=False,
                portal_offer_pending_client=True,
                portal_offer_id=(
                    int(result.offer.id) if result.offer is not None else None
                ),
            )

        # --- Legacy / non-PORTAL : assignation immédiate ---
        if client is not None and is_portal_client(client):
            debtor_user_id = resolve_portal_booking_debtor_user_id(booking)
            if debtor_user_id is not None:
                hold = resolve_portal_payment_hold(int(debtor_user_id), int(company_id))
                if hold.is_hold:
                    return AcceptReservationResult(
                        ok=False,
                        error={
                            "error": ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
                            "message": (
                                "Ce client a une facture échue auprès de votre "
                                "entreprise. L'acceptation est refusée."
                            ),
                        },
                        status_code=409,
                    )

        booking.company_id = company_id
        _set_status(booking, "accepted")
        return AcceptReservationResult(ok=True, should_trigger_dispatch=True)
