"""Rappels / mise en demeure PORTAL — au nom du transporteur créancier.

Aucun frais, intérêt ni poursuite automatique. Le hold 6D reste inchangé.
FORMAL_NOTICE exige une validation explicite du créancier.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from ext import db
from models.booking import Booking
from models.client_booking_contract_event import (
    EVENT_BOOKING_CANCELLED,
    EVENT_BOOKING_CREATED,
    EVENT_BOOKING_MODIFIED,
    ClientBookingContractEvent,
)
from models.company import Company
from models.portal_receivable import (
    DISPUTE_OPEN,
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_PAID,
    PortalReceivable,
    PortalReceivableDispute,
)
from models.portal_receivable_dunning import (
    CHANNEL_EMAIL,
    CHANNEL_INTERNAL,
    CHANNEL_LETTER_DRAFT,
    DEFAULT_FIRST_REMINDER_DAYS,
    DEFAULT_FORMAL_NOTICE_DAYS,
    DEFAULT_SECOND_REMINDER_DAYS,
    DELIVERY_DRAFT,
    DELIVERY_RECORDED,
    DELIVERY_SENT,
    DUNNING_COLLECTION_PREPARED,
    DUNNING_FORMAL_NOTICE,
    DUNNING_REMINDER_1,
    DUNNING_REMINDER_2,
    PORTAL_DUNNING_TEMPLATE_VERSION,
    PortalReceivableDunningEvent,
    PortalReceivableDunningPolicy,
)
from services.billing.portal_payment_hold import (
    business_calendar_date,
    current_business_date,
)
from services.billing.portal_receivable import PortalReceivableError

EmailSender = Callable[..., dict[str, Any]]

REASON_NOT_OVERDUE = "not_overdue"
REASON_PAID = "paid"
REASON_CANCELLED = "cancelled"
REASON_DISPUTED = "disputed"
REASON_FORMAL_NOTICE_MISSING = "formal_notice_missing"
REASON_DEBTOR_NAME_MISSING = "debtor_name_missing"
REASON_DEBTOR_ADDRESS_MISSING = "debtor_address_missing"
REASON_CREDITOR_IDENTITY_MISSING = "creditor_identity_missing"
REASON_INVOICE_REFERENCE_MISSING = "invoice_reference_missing"
REASON_COLLECTION_ALREADY_PREPARED = "collection_already_prepared"

READY = "ready"
NOT_READY = "not_ready"


@dataclass(frozen=True, slots=True)
class DunningEligibility:
    eligible: bool
    reason: str | None
    days_past_due: int
    next_event_type: str | None
    policy: PortalReceivableDunningPolicy | None


@dataclass(frozen=True, slots=True)
class CollectionReadiness:
    state: str
    reasons: tuple[str, ...]
    dossier: dict[str, Any] | None = None

    @property
    def is_ready(self) -> bool:
        return self.state == READY


def get_or_create_dunning_policy(company_id: int) -> PortalReceivableDunningPolicy:
    row = PortalReceivableDunningPolicy.query.filter_by(
        company_id=int(company_id)
    ).one_or_none()
    if row is not None:
        return row
    row = PortalReceivableDunningPolicy(
        company_id=int(company_id),
        enabled=True,
        first_reminder_days=DEFAULT_FIRST_REMINDER_DAYS,
        second_reminder_days=DEFAULT_SECOND_REMINDER_DAYS,
        formal_notice_days=DEFAULT_FORMAL_NOTICE_DAYS,
        charge_default_interest=False,
        default_interest_rate=None,
    )
    db.session.add(row)
    db.session.flush()
    return row


def update_dunning_policy(
    *,
    company_id: int,
    first_reminder_days: int | None = None,
    second_reminder_days: int | None = None,
    formal_notice_days: int | None = None,
    enabled: bool | None = None,
) -> PortalReceivableDunningPolicy:
    policy = get_or_create_dunning_policy(company_id)
    if first_reminder_days is not None:
        policy.first_reminder_days = int(first_reminder_days)
    if second_reminder_days is not None:
        policy.second_reminder_days = int(second_reminder_days)
    if formal_notice_days is not None:
        policy.formal_notice_days = int(formal_notice_days)
    if enabled is not None:
        policy.enabled = bool(enabled)
    if policy.second_reminder_days < policy.first_reminder_days:
        raise PortalReceivableError(
            "second_reminder_days doit être ≥ first_reminder_days.",
            code="dunning_policy_invalid",
        )
    if policy.formal_notice_days < policy.second_reminder_days:
        raise PortalReceivableError(
            "formal_notice_days doit être ≥ second_reminder_days.",
            code="dunning_policy_invalid",
        )
    db.session.flush()
    return policy


def serialize_dunning_policy(policy: PortalReceivableDunningPolicy) -> dict[str, Any]:
    return {
        "company_id": policy.company_id,
        "enabled": bool(policy.enabled),
        "first_reminder_days": int(policy.first_reminder_days),
        "second_reminder_days": int(policy.second_reminder_days),
        "formal_notice_days": int(policy.formal_notice_days),
        "charge_default_interest": False,
        "default_interest_rate": None,
        "reminder_fees": "NOT_IMPLEMENTED",
        "auto_pursuit": False,
        "formal_notice_auto_sent": False,
        "creditor_approval_required_for_formal_notice": True,
        "reminders_automatable": True,
    }


def _terminal_dunning_reason(
    receivable: PortalReceivable, *, as_of_date: date
) -> str | None:
    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        return REASON_CANCELLED
    if receivable.status == RECEIVABLE_DISPUTED or receivable.disputed_at is not None:
        return REASON_DISPUTED
    if (
        receivable.status == RECEIVABLE_PAID
        or Decimal(str(receivable.balance_due)) <= 0
    ):
        return REASON_PAID
    due = business_calendar_date(receivable.due_date)
    if due is None or due >= as_of_date:
        return REASON_NOT_OVERDUE
    return None


def receivable_eligible_for_dunning(
    receivable: PortalReceivable, *, as_of: date | datetime | None = None
) -> bool:
    return (
        _terminal_dunning_reason(receivable, as_of_date=current_business_date(as_of))
        is None
    )


def _days_past_due(receivable: PortalReceivable, *, as_of_date: date) -> int:
    due = business_calendar_date(receivable.due_date)
    if due is None:
        return 0
    return max(0, (as_of_date - due).days)


def _successful_events(receivable_id: int) -> list[PortalReceivableDunningEvent]:
    return (
        PortalReceivableDunningEvent.query.filter_by(receivable_id=int(receivable_id))
        .filter(
            PortalReceivableDunningEvent.delivery_status.in_(
                (DELIVERY_SENT, DELIVERY_RECORDED, DELIVERY_DRAFT)
            )
        )
        .order_by(PortalReceivableDunningEvent.id.asc())
        .all()
    )


def _successful_event_types(receivable_id: int) -> set[str]:
    return {str(r.event_type) for r in _successful_events(receivable_id)}


def resolve_dunning_eligibility(
    receivable: PortalReceivable,
    *,
    as_of: date | datetime | None = None,
) -> DunningEligibility:
    policy = get_or_create_dunning_policy(int(receivable.creditor_company_id))
    as_of_date = current_business_date(as_of)
    terminal = _terminal_dunning_reason(receivable, as_of_date=as_of_date)
    if terminal is not None:
        return DunningEligibility(
            eligible=False,
            reason=terminal,
            days_past_due=_days_past_due(receivable, as_of_date=as_of_date),
            next_event_type=None,
            policy=policy,
        )
    if not policy.enabled:
        return DunningEligibility(
            eligible=False,
            reason="policy_disabled",
            days_past_due=_days_past_due(receivable, as_of_date=as_of_date),
            next_event_type=None,
            policy=policy,
        )
    days = _days_past_due(receivable, as_of_date=as_of_date)
    done = _successful_event_types(int(receivable.id))
    if DUNNING_COLLECTION_PREPARED in done:
        return DunningEligibility(
            eligible=False,
            reason=REASON_COLLECTION_ALREADY_PREPARED,
            days_past_due=days,
            next_event_type=None,
            policy=policy,
        )
    if DUNNING_FORMAL_NOTICE in done:
        return DunningEligibility(
            eligible=True,
            reason=None,
            days_past_due=days,
            next_event_type=DUNNING_COLLECTION_PREPARED,
            policy=policy,
        )
    if DUNNING_REMINDER_2 in done:
        if days >= int(policy.formal_notice_days):
            return DunningEligibility(
                eligible=True,
                reason=None,
                days_past_due=days,
                next_event_type=DUNNING_FORMAL_NOTICE,
                policy=policy,
            )
        return DunningEligibility(
            eligible=False,
            reason="waiting_formal_notice_window",
            days_past_due=days,
            next_event_type=None,
            policy=policy,
        )
    if DUNNING_REMINDER_1 in done:
        if days >= int(policy.second_reminder_days):
            return DunningEligibility(
                eligible=True,
                reason=None,
                days_past_due=days,
                next_event_type=DUNNING_REMINDER_2,
                policy=policy,
            )
        return DunningEligibility(
            eligible=False,
            reason="waiting_second_reminder_window",
            days_past_due=days,
            next_event_type=None,
            policy=policy,
        )
    if days >= int(policy.first_reminder_days):
        return DunningEligibility(
            eligible=True,
            reason=None,
            days_past_due=days,
            next_event_type=DUNNING_REMINDER_1,
            policy=policy,
        )
    return DunningEligibility(
        eligible=False,
        reason="waiting_first_reminder_window",
        days_past_due=days,
        next_event_type=None,
        policy=policy,
    )


def _money(value: object) -> Decimal:
    return Decimal(str(value or "0"))


def _format_chf(amount: Decimal) -> str:
    return f"{_money(amount):.2f} CHF"


def _format_date(value: datetime | None) -> str:
    if value is None:
        return "—"
    d = business_calendar_date(value)
    if d is None:
        return "—"
    return d.strftime("%d.%m.%Y")


def _nonempty(value: object | None) -> bool:
    return bool(str(value or "").strip())


def _creditor_postal_address(company: Company) -> str | None:
    line1 = getattr(company, "domicile_address_line1", None)
    zip_c = getattr(company, "domicile_zip", None)
    city = getattr(company, "domicile_city", None)
    if _nonempty(line1) and (_nonempty(zip_c) or _nonempty(city)):
        parts = [str(line1).strip()]
        line2 = getattr(company, "domicile_address_line2", None)
        if _nonempty(line2):
            parts.append(str(line2).strip())
        loc = " ".join(
            p for p in [str(zip_c or "").strip(), str(city or "").strip()] if p
        )
        if loc:
            parts.append(loc)
        return ", ".join(parts)
    addr = getattr(company, "address", None)
    return str(addr).strip() if _nonempty(addr) else None


def creditor_identity_complete(company: Company) -> bool:
    return _nonempty(getattr(company, "name", None)) and _nonempty(
        _creditor_postal_address(company)
    )


def build_claim_title(receivable: PortalReceivable) -> str:
    dates: list[str] = []
    for line in receivable.lines or []:
        booking = db.session.get(Booking, int(line.booking_id))
        if booking is not None and getattr(booking, "scheduled_time", None):
            dates.append(_format_date(booking.scheduled_time))
    dates_txt = ", ".join(dates) if dates else "dates concernées"
    return (
        f"Facture n° {receivable.external_invoice_number} relative aux "
        f"prestations de transport des {dates_txt}"
    )


def build_collection_dossier(receivable: PortalReceivable) -> dict[str, Any]:
    company = db.session.get(Company, int(receivable.creditor_company_id))
    lines_payload = []
    booking_ids: list[int] = []
    contract_events: list[dict[str, Any]] = []
    for line in receivable.lines or []:
        booking_ids.append(int(line.booking_id))
        lines_payload.append(
            {
                "booking_id": line.booking_id,
                "booking_contract_event_id": line.booking_contract_event_id,
                "invoiced_amount": float(line.invoiced_amount),
                "description": line.description,
            }
        )
        events = (
            ClientBookingContractEvent.query.filter_by(booking_id=int(line.booking_id))
            .filter(
                ClientBookingContractEvent.event_type.in_(
                    (
                        EVENT_BOOKING_CREATED,
                        EVENT_BOOKING_MODIFIED,
                        EVENT_BOOKING_CANCELLED,
                    )
                )
            )
            .order_by(ClientBookingContractEvent.id.asc())
            .all()
        )
        for ev in events:
            contract_events.append(
                {
                    "id": ev.id,
                    "booking_id": ev.booking_id,
                    "event_type": ev.event_type,
                    "occurred_at": (
                        ev.occurred_at.isoformat()
                        if getattr(ev, "occurred_at", None)
                        else None
                    ),
                }
            )
    disputes = [
        {
            "id": d.id,
            "status": d.status,
            "reason": d.reason,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in (receivable.disputes or [])
    ]
    dunning = [
        serialize_dunning_event(e) for e in _successful_events(int(receivable.id))
    ]
    formal = next(
        (
            e
            for e in reversed(_successful_events(int(receivable.id)))
            if e.event_type == DUNNING_FORMAL_NOTICE
        ),
        None,
    )
    return {
        "receivable_id": receivable.id,
        "claim_title": build_claim_title(receivable),
        "creditor": {
            "company_id": receivable.creditor_company_id,
            "display_name": receivable.creditor_name_snapshot,
            "legal_form": getattr(company, "legal_form", None) if company else None,
            "uid_ide": getattr(company, "uid_ide", None) if company else None,
            "postal_address": (_creditor_postal_address(company) if company else None),
            "contact_email": (
                (
                    getattr(company, "billing_email", None)
                    or getattr(company, "contact_email", None)
                )
                if company
                else None
            ),
            "contact_phone": (
                getattr(company, "contact_phone", None) if company else None
            ),
            "iban_present": bool(getattr(company, "iban", None)) if company else False,
        },
        "debtor": {
            "user_id": receivable.debtor_user_id,
            "name": receivable.debtor_name_snapshot,
            "email": receivable.debtor_email_snapshot,
            "phone": receivable.debtor_phone_snapshot,
            "postal_address": receivable.debtor_billing_address_snapshot,
        },
        "invoice": {
            "external_invoice_number": receivable.external_invoice_number,
            "issued_at": (
                receivable.issued_at.isoformat() if receivable.issued_at else None
            ),
            "due_date": receivable.due_date.isoformat()
            if receivable.due_date
            else None,
            "currency": receivable.currency,
            "principal_initial": float(receivable.total_amount),
            "amount_paid": float(receivable.amount_paid),
            "balance_due": float(receivable.balance_due),
        },
        "lines": lines_payload,
        "booking_ids": booking_ids,
        "contract_events": contract_events,
        "payments": [
            {
                "amount": float(p.amount),
                "paid_at": p.paid_at.isoformat() if p.paid_at else None,
                "method": p.method,
                "reference": p.reference,
            }
            for p in (receivable.payments or [])
        ],
        "dunning_history": dunning,
        "dispute_history": disputes,
        "formal_notice_event_id": formal.id if formal else None,
        "availability": {
            "debtor_legal_name": (
                "AVAILABLE" if _nonempty(receivable.debtor_name_snapshot) else "MISSING"
            ),
            "debtor_postal_address": (
                "AVAILABLE"
                if _nonempty(receivable.debtor_billing_address_snapshot)
                else "MISSING"
            ),
            "creditor_legal_identity": (
                "AVAILABLE"
                if company and creditor_identity_complete(company)
                else "PARTIAL"
                if company and _nonempty(company.name)
                else "MISSING"
            ),
            "title_reason_of_claim": "AVAILABLE",
            "invoice_reference": (
                "AVAILABLE"
                if _nonempty(receivable.external_invoice_number)
                else "MISSING"
            ),
        },
    }


def resolve_portal_collection_readiness(
    receivable_id: int,
    *,
    as_of: date | datetime | None = None,
) -> CollectionReadiness:
    receivable = db.session.get(PortalReceivable, int(receivable_id))
    if receivable is None:
        return CollectionReadiness(
            state=NOT_READY, reasons=("receivable_not_found",), dossier=None
        )
    as_of_date = current_business_date(as_of)
    reasons: list[str] = []
    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        reasons.append(REASON_CANCELLED)
    if receivable.status == RECEIVABLE_DISPUTED or receivable.disputed_at is not None:
        open_d = PortalReceivableDispute.query.filter_by(
            receivable_id=int(receivable.id), status=DISPUTE_OPEN
        ).first()
        if open_d is not None or receivable.disputed_at is not None:
            reasons.append(REASON_DISPUTED)
    if receivable.status == RECEIVABLE_PAID or _money(receivable.balance_due) <= 0:
        reasons.append(REASON_PAID)
    due = business_calendar_date(receivable.due_date)
    if due is None or due >= as_of_date:
        reasons.append(REASON_NOT_OVERDUE)
    done = _successful_event_types(int(receivable.id))
    if DUNNING_FORMAL_NOTICE not in done:
        reasons.append(REASON_FORMAL_NOTICE_MISSING)
    if not _nonempty(receivable.debtor_name_snapshot):
        reasons.append(REASON_DEBTOR_NAME_MISSING)
    if not _nonempty(receivable.debtor_billing_address_snapshot):
        reasons.append(REASON_DEBTOR_ADDRESS_MISSING)
    company = db.session.get(Company, int(receivable.creditor_company_id))
    if company is None or not creditor_identity_complete(company):
        reasons.append(REASON_CREDITOR_IDENTITY_MISSING)
    if not _nonempty(receivable.external_invoice_number):
        reasons.append(REASON_INVOICE_REFERENCE_MISSING)

    dossier = build_collection_dossier(receivable)
    # Dédupliquer en conservant l'ordre
    uniq: list[str] = []
    for r in reasons:
        if r not in uniq:
            uniq.append(r)
    if uniq:
        return CollectionReadiness(
            state=NOT_READY, reasons=tuple(uniq), dossier=dossier
        )
    return CollectionReadiness(state=READY, reasons=(), dossier=dossier)


def render_dunning_message(
    *,
    receivable: PortalReceivable,
    company: Company,
    event_type: str,
    payment_deadline: date | None = None,
) -> tuple[str, str]:
    """Sujet + corps texte — au nom du créancier, pas de LIRIE créancier."""
    creditor = str(receivable.creditor_name_snapshot or company.name)
    debtor = str(receivable.debtor_name_snapshot or "Client")
    invoice_no = str(receivable.external_invoice_number)
    payments = list(receivable.payments or [])
    paid_lines = (
        "\n".join(
            f"  - {_format_date(p.paid_at)} : {_format_chf(_money(p.amount))}"
            f" ({p.method})"
            for p in payments
        )
        or "  - Aucun paiement enregistré"
    )
    if event_type == DUNNING_REMINDER_1:
        title = "Rappel de paiement"
    elif event_type == DUNNING_REMINDER_2:
        title = "Deuxième rappel de paiement"
    elif event_type == DUNNING_FORMAL_NOTICE:
        title = "Mise en demeure"
    else:
        title = "Dossier de recouvrement"

    subject = f"{creditor} — {title} facture {invoice_no}"
    creditor_addr = _creditor_postal_address(company) or "—"
    payment_coords = []
    if getattr(company, "iban", None):
        payment_coords.append(f"IBAN : {company.iban}")
    if getattr(company, "billing_email", None) or getattr(
        company, "contact_email", None
    ):
        payment_coords.append(
            "Contact : "
            + str(
                getattr(company, "billing_email", None)
                or getattr(company, "contact_email", None)
            )
        )
    payment_block = (
        "\n".join(f"  - {c}" for c in payment_coords)
        if payment_coords
        else "  - Coordonnées de paiement à obtenir auprès du créancier"
    )
    deadline = payment_deadline or (current_business_date() + timedelta(days=10))
    deadline_txt = deadline.strftime("%d.%m.%Y")

    body = (
        f"{title}\n"
        f"\n"
        f"Émetteur (créancier) : {creditor}\n"
        f"Adresse créancier : {creditor_addr}\n"
        f"Destinataire (débiteur) : {debtor}\n"
        f"\n"
        f"Facture n° {invoice_no}\n"
        f"Date de facture : {_format_date(receivable.issued_at)}\n"
        f"Date d'échéance initiale : {_format_date(receivable.due_date)}\n"
        f"Montant initial : {_format_chf(_money(receivable.total_amount))}\n"
        f"Paiements reçus :\n{paid_lines}\n"
        f"Principal / solde restant dû : {_format_chf(_money(receivable.balance_due))}\n"
        f"\n"
        f"Merci de régulariser le solde auprès de {creditor}.\n"
        f"\n"
        f"—\n"
        f"Ce message est envoyé pour le compte de {creditor}.\n"
        f"Lirie fournit uniquement l'infrastructure technique.\n"
    )
    if event_type == DUNNING_FORMAL_NOTICE:
        body += (
            f"\nNouvelle date limite de règlement : {deadline_txt}\n"
            f"Coordonnées de paiement :\n{payment_block}\n"
            f"\n"
            f"En l'absence de règlement dans ce délai, le créancier pourra "
            f"préparer un dossier de recouvrement et décider des suites "
            f"appropriées. Aucune poursuite n'est déposée automatiquement "
            f"par la plateforme.\n"
        )
    if event_type == DUNNING_COLLECTION_PREPARED:
        body += (
            "\nDossier prêt pour le créancier. La réquisition de poursuite "
            "reste une action explicite du transporteur.\n"
        )
    return subject, body


def _hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _default_email_sender(**kwargs: Any) -> dict[str, Any]:
    from services.notifications.email import send_email_notification

    return send_email_notification(**kwargs)


def _creditor_email_identity(company: Company) -> tuple[str | None, str, str | None]:
    from_name = str(getattr(company, "name", "") or "Transporteur")
    reply = (
        getattr(company, "billing_email", None)
        or getattr(company, "contact_email", None)
        or None
    )
    reply_s = str(reply).strip() if reply else None
    return None, from_name, reply_s


def emit_portal_dunning_event(
    *,
    receivable: PortalReceivable,
    event_type: str,
    initiated_by_user_id: int | None,
    channel: str = CHANNEL_EMAIL,
    as_of: date | datetime | None = None,
    email_sender: EmailSender | None = None,
    force: bool = False,
    creditor_approved: bool = False,
) -> PortalReceivableDunningEvent:
    """Émet un événement. FORMAL_NOTICE exige creditor_approved=True."""
    company = db.session.get(Company, int(receivable.creditor_company_id))
    if company is None:
        raise PortalReceivableError(
            "Créancier introuvable.",
            code="creditor_not_found",
        )
    eligibility = resolve_dunning_eligibility(receivable, as_of=as_of)
    if not force and (
        not eligibility.eligible or eligibility.next_event_type != event_type
    ):
        code = "dunning_not_due"
        if eligibility.reason in (
            REASON_DISPUTED,
            REASON_PAID,
            REASON_CANCELLED,
            REASON_NOT_OVERDUE,
        ):
            code = f"dunning_blocked_{eligibility.reason}"
        raise PortalReceivableError(
            "Cette étape de rappel n'est pas encore due ou déjà effectuée.",
            code=code,
        )
    if event_type == DUNNING_FORMAL_NOTICE and not creditor_approved:
        raise PortalReceivableError(
            "La mise en demeure nécessite une validation explicite du créancier.",
            code="formal_notice_approval_required",
        )

    dossier_json: str | None = None
    dossier_hash: str | None = None
    formal_notice_event_id: int | None = None

    if event_type == DUNNING_COLLECTION_PREPARED:
        readiness = resolve_portal_collection_readiness(int(receivable.id), as_of=as_of)
        if not readiness.is_ready:
            raise PortalReceivableError(
                "Dossier de recouvrement incomplet : " + ", ".join(readiness.reasons),
                code="collection_not_ready",
            )
        dossier = readiness.dossier or {}
        dossier["prepared_at"] = datetime.now(UTC).isoformat()
        dossier["prepared_by_user_id"] = initiated_by_user_id
        dossier_json = json.dumps(dossier, ensure_ascii=False, sort_keys=True)
        dossier_hash = _hash_body(dossier_json)
        formal_notice_event_id = dossier.get("formal_notice_event_id")

    subject, body = render_dunning_message(
        receivable=receivable, company=company, event_type=event_type
    )
    body_hash = _hash_body(body)
    recipient_email = (receivable.debtor_email_snapshot or "").strip() or None

    delivery_status = DELIVERY_RECORDED
    provider_message_id: str | None = None
    used_channel = channel

    if event_type == DUNNING_COLLECTION_PREPARED:
        used_channel = CHANNEL_INTERNAL
        delivery_status = DELIVERY_RECORDED
    elif channel == CHANNEL_LETTER_DRAFT:
        used_channel = CHANNEL_LETTER_DRAFT
        delivery_status = DELIVERY_DRAFT
    elif channel == CHANNEL_EMAIL:
        if not recipient_email:
            raise PortalReceivableError(
                "Aucune adresse e-mail débiteur pour envoyer le rappel.",
                code="debtor_email_missing",
            )
        _from_email, from_name, reply_to = _creditor_email_identity(company)
        sender = email_sender or _default_email_sender
        result = sender(
            email=recipient_email,
            subject=subject,
            body=body,
            notification_type=f"portal_dunning_{event_type.lower()}",
            html=False,
            from_name=from_name,
            reply_to=reply_to,
        )
        if not result.get("ok"):
            raise PortalReceivableError(
                "Échec d'envoi e-mail — rappel non marqué comme envoyé.",
                code="dunning_email_failed",
            )
        delivery_status = DELIVERY_SENT
        provider_message_id = (
            str(result.get("message_id")) if result.get("message_id") else None
        )
    else:
        raise PortalReceivableError(
            "Canal non supporté.",
            code="dunning_channel_invalid",
        )

    event = PortalReceivableDunningEvent(
        receivable_id=int(receivable.id),
        creditor_company_id=int(receivable.creditor_company_id),
        debtor_user_id=int(receivable.debtor_user_id),
        event_type=event_type,
        occurred_at=datetime.now(UTC),
        channel=used_channel,
        recipient=recipient_email,
        recipient_email=recipient_email,
        template_version=PORTAL_DUNNING_TEMPLATE_VERSION,
        rendered_subject=subject,
        rendered_body=body,
        rendered_body_hash=body_hash,
        balance_due_snapshot=_money(receivable.balance_due),
        due_date_snapshot=receivable.due_date,
        external_invoice_number_snapshot=str(receivable.external_invoice_number),
        delivery_status=delivery_status,
        provider_message_id=provider_message_id,
        delivery_error=None,
        initiated_by_user_id=initiated_by_user_id,
        dossier_snapshot=dossier_json,
        dossier_snapshot_hash=dossier_hash,
        formal_notice_event_id=formal_notice_event_id,
    )
    db.session.add(event)
    db.session.flush()
    return event


def serialize_dunning_event(
    event: PortalReceivableDunningEvent, *, for_client: bool = False
) -> dict[str, Any]:
    base = {
        "id": event.id,
        "receivable_id": event.receivable_id,
        "event_type": event.event_type,
        "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
        "channel": event.channel,
        "delivery_status": event.delivery_status,
        "balance_due_snapshot": float(event.balance_due_snapshot),
        "external_invoice_number_snapshot": event.external_invoice_number_snapshot,
    }
    if for_client:
        return base
    base.update(
        {
            "creditor_company_id": event.creditor_company_id,
            "debtor_user_id": event.debtor_user_id,
            "recipient": event.recipient,
            "recipient_email": event.recipient_email,
            "template_version": event.template_version,
            "rendered_subject": event.rendered_subject,
            "rendered_body_hash": event.rendered_body_hash,
            "due_date_snapshot": (
                event.due_date_snapshot.isoformat() if event.due_date_snapshot else None
            ),
            "provider_message_id": event.provider_message_id,
            "dossier_snapshot_hash": event.dossier_snapshot_hash,
            "formal_notice_event_id": event.formal_notice_event_id,
        }
    )
    return base


def serialize_collection_readiness(result: CollectionReadiness) -> dict[str, Any]:
    return {
        "state": result.state,
        "reasons": list(result.reasons),
        "dossier": result.dossier,
    }
