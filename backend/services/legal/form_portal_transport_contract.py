"""Formation atomique du contrat PORTAL — conditional_order_v1 (7B.5)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from ext import db
from models.portal_client_conditional_order import PortalClientConditionalOrder
from models.portal_transport_contract_formed import PortalTransportContractFormed
from services.billing.portal_booking_debtor import resolve_portal_booking_debtor_user_id
from services.billing.portal_payment_hold import (
    ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
    resolve_portal_payment_hold,
)
from services.legal.portal_cancellation_policy import (
    get_current_cancellation_policy,
    load_company_cancellation_config,
)
from services.legal.portal_channel_cancellation_caps import (
    get_current_channel_cancellation_policy,
    validate_company_policy_against_channel,
)
from services.legal.portal_double_validation import (
    ERROR_CARRIER_NOT_IN_ORDER_POOL,
    ERROR_PORTAL_NO_CANCELLATION_POLICY,
    ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT,
    ERROR_TRANSPORT_ALREADY_ASSIGNED,
    FLOW_CONDITIONAL_ORDER_V1,
    booking_uses_conditional_order,
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
class FormTransportContractResult:
    ok: bool
    contract: PortalTransportContractFormed | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None
    # True si même company rejoue l'accept (idempotent).
    idempotent_replay: bool = False


def _pool_company_ids(order: PortalClientConditionalOrder) -> set[int]:
    snap = order.eligible_carriers_snapshot or []
    ids: set[int] = set()
    if not isinstance(snap, list):
        return ids
    for row in snap:
        if not isinstance(row, dict):
            continue
        try:
            ids.add(int(row["company_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def form_portal_transport_contract_on_accept(
    *,
    booking: Any,
    company_id: int,
    offered_amount: Any | None = None,
    actor_user_id: int | None = None,
) -> FormTransportContractResult:
    """1er transporteur valide du pool figé → TRANSPORT_CONTRACT_FORMED.

    Idempotent pour la même company. Concurrent autre company →
    transport_already_assigned.
    """
    if not booking_uses_conditional_order(booking):
        return FormTransportContractResult(
            ok=False,
            error={"error": "portal_conditional_order_inactive"},
            status_code=400,
        )

    from models import Company
    from models.booking import Booking
    from services.pricing.portal_carrier_ceiling import (
        estimate_portal_carrier_offer_amount,
    )

    locked = (
        db.session.query(Booking)
        .filter_by(id=int(booking.id))
        .with_for_update()
        .one()
    )

    existing = PortalTransportContractFormed.query.filter_by(
        booking_id=int(locked.id)
    ).one_or_none()
    if existing is not None:
        if int(existing.company_id) == int(company_id):
            return FormTransportContractResult(
                ok=True, contract=existing, idempotent_replay=True
            )
        return FormTransportContractResult(
            ok=False,
            error={
                "error": ERROR_TRANSPORT_ALREADY_ASSIGNED,
                "message": "Ce transport a déjà été accepté par une autre entreprise.",
            },
            status_code=409,
        )

    # Déjà assigné sans preuve (incohérence) → même erreur pour autre company.
    assigned = getattr(locked, "company_id", None)
    if assigned is not None and int(assigned) > 0:
        if int(assigned) == int(company_id):
            # Pas de preuve — ne pas inventer ; refuse soft
            return FormTransportContractResult(
                ok=False,
                error={"error": "portal_contract_evidence_missing"},
                status_code=409,
            )
        return FormTransportContractResult(
            ok=False,
            error={
                "error": ERROR_TRANSPORT_ALREADY_ASSIGNED,
                "message": "Ce transport a déjà été accepté par une autre entreprise.",
            },
            status_code=409,
        )

    status_str = str(
        getattr(getattr(locked, "status", None), "value", getattr(locked, "status", ""))
        or ""
    ).upper()
    if status_str != "PENDING":
        return FormTransportContractResult(
            ok=False,
            error={"error": "reservation_not_acceptable"},
            status_code=400,
        )

    order = (
        PortalClientConditionalOrder.query.filter_by(booking_id=int(locked.id))
        .with_for_update()
        .one_or_none()
    )
    if order is None:
        return FormTransportContractResult(
            ok=False,
            error={"error": "portal_conditional_order_missing"},
            status_code=409,
        )

    if int(company_id) not in _pool_company_ids(order):
        return FormTransportContractResult(
            ok=False,
            error={
                "error": ERROR_CARRIER_NOT_IN_ORDER_POOL,
                "message": (
                    "Votre entreprise ne fait pas partie du pool "
                    "figé pour cette commande."
                ),
            },
            status_code=409,
        )

    company = Company.query.get(int(company_id))
    if not company or not company.is_approved:
        return FormTransportContractResult(
            ok=False,
            error={"error": "Entreprise non approuvée"},
            status_code=403,
        )

    debtor_user_id = resolve_portal_booking_debtor_user_id(locked)
    if debtor_user_id is not None:
        hold = resolve_portal_payment_hold(int(debtor_user_id), int(company_id))
        if hold.is_hold:
            return FormTransportContractResult(
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

    amount = offered_amount
    if amount is None:
        amount = estimate_portal_carrier_offer_amount(locked, int(company_id))
    if amount is None:
        return FormTransportContractResult(
            ok=False,
            error={
                "error": "portal_carrier_quote_unavailable",
                "message": (
                    "Impossible de calculer votre tarif pour cette course."
                ),
            },
            status_code=409,
        )

    try:
        quote = _money(amount)
    except Exception:
        return FormTransportContractResult(
            ok=False,
            error={"error": "invalid_offer_amount"},
            status_code=400,
        )
    if quote <= Decimal("0.00"):
        return FormTransportContractResult(
            ok=False,
            error={"error": "invalid_offer_amount"},
            status_code=400,
        )

    ceiling = _money(order.client_ceiling)
    if quote > ceiling:
        return FormTransportContractResult(
            ok=False,
            error={
                "error": ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT,
                "message": (
                    "Le prix proposé dépasse le plafond accepté par le client."
                ),
            },
            status_code=409,
        )

    company_policy_row = get_current_cancellation_policy(int(company_id))
    if company_policy_row is None:
        return FormTransportContractResult(
            ok=False,
            error={
                "error": ERROR_PORTAL_NO_CANCELLATION_POLICY,
                "message": "Publiez d'abord vos conditions d'annulation PORTAL.",
            },
            status_code=409,
        )

    channel = get_current_channel_cancellation_policy()
    if channel is None:
        return FormTransportContractResult(
            ok=False,
            error={
                "error": "portal_channel_cancellation_policy_required",
                "message": "Politique canal LIRIE d'annulation absente.",
            },
            status_code=409,
        )

    # Hash/version canal figés sur la commande doivent encore matcher le courant
    # OU on utilise le snapshot de commande pour la preuve (préféré).
    channel_version = str(order.channel_policy_version)
    channel_hash = str(order.channel_policy_hash)
    channel_snapshot = str(order.channel_policy_snapshot)
    channel_body = channel.body_json if isinstance(channel.body_json, dict) else {}
    # Si la version commande ≠ courant : on valide quand même contre le body
    # courant pour conformité, mais on fige le snapshot de la commande.
    config = load_company_cancellation_config(int(company_id))
    cap_check = validate_company_policy_against_channel(
        company_policy=config if isinstance(config, dict) else None,
        channel_body=channel_body,
    )
    if not cap_check.ok:
        return FormTransportContractResult(
            ok=False,
            error={
                "error": str(cap_check.error or "portal_cancellation_cap_rejected"),
                "message": str(
                    cap_check.message
                    or "Politique d'annulation hors plafonds canal LIRIE."
                ),
            },
            status_code=409,
        )

    legal_name = str(
        getattr(company, "legal_name", None)
        or getattr(company, "name", None)
        or f"Entreprise #{company_id}"
    )

    contract = PortalTransportContractFormed(
        booking_id=int(locked.id),
        company_id=int(company_id),
        carrier_legal_name=legal_name,
        carrier_quote=quote,
        client_ceiling=ceiling,
        currency="CHF",
        flow_version=FLOW_CONDITIONAL_ORDER_V1,
        company_policy_version=str(company_policy_row.version),
        company_policy_hash=str(company_policy_row.content_hash),
        company_policy_snapshot=str(company_policy_row.body_text),
        channel_policy_version=channel_version,
        channel_policy_hash=channel_hash,
        channel_policy_snapshot=channel_snapshot,
        terms_of_service_version=str(order.terms_of_service_version),
        terms_of_service_hash=str(order.terms_of_service_hash),
        transport_terms_version=str(order.transport_terms_version),
        transport_terms_hash=str(order.transport_terms_hash),
        actor_user_id=int(actor_user_id) if actor_user_id else None,
        formed_at=now_utc(),
    )
    db.session.add(contract)

    locked.company_id = int(company_id)
    # Prix affiché / legacy = quote contractuelle (pas estimation).
    locked.amount = float(quote)
    _set_status(locked, "accepted")

    try:
        from services.billing.billing_party_linker import (
            ensure_patient_destination_billing_party,
        )

        ensure_patient_destination_billing_party(locked)
    except Exception:
        pass

    db.session.flush()
    return FormTransportContractResult(ok=True, contract=contract)
