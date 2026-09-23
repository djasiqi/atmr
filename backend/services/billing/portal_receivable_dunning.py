"""Rappels / mise en demeure PORTAL — au nom du transporteur créancier.

Aucun frais, intérêt ni poursuite automatique. Le hold 6D reste inchangé.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from ext import db
from models.company import Company
from models.portal_receivable import (
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_PAID,
    PortalReceivable,
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


@dataclass(frozen=True, slots=True)
class DunningEligibility:
    eligible: bool
    reason: str | None
    days_past_due: int
    next_event_type: str | None
    policy: PortalReceivableDunningPolicy | None


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
    }


def receivable_eligible_for_dunning(
    receivable: PortalReceivable, *, as_of: date | datetime | None = None
) -> bool:
    as_of_date = current_business_date(as_of)
    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        return False
    if receivable.status == RECEIVABLE_DISPUTED or receivable.disputed_at is not None:
        return False
    if receivable.status == RECEIVABLE_PAID or Decimal(str(receivable.balance_due)) <= 0:
        return False
    due = business_calendar_date(receivable.due_date)
    return due is not None and due < as_of_date


def _days_past_due(receivable: PortalReceivable, *, as_of_date: date) -> int:
    due = business_calendar_date(receivable.due_date)
    if due is None:
        return 0
    return max(0, (as_of_date - due).days)


def _successful_event_types(receivable_id: int) -> set[str]:
    rows = (
        PortalReceivableDunningEvent.query.filter_by(receivable_id=int(receivable_id))
        .filter(
            PortalReceivableDunningEvent.delivery_status.in_(
                (DELIVERY_SENT, DELIVERY_RECORDED, DELIVERY_DRAFT)
            )
        )
        .all()
    )
    return {str(r.event_type) for r in rows}


def resolve_dunning_eligibility(
    receivable: PortalReceivable,
    *,
    as_of: date | datetime | None = None,
) -> DunningEligibility:
    policy = get_or_create_dunning_policy(int(receivable.creditor_company_id))
    as_of_date = current_business_date(as_of)
    if not receivable_eligible_for_dunning(receivable, as_of=as_of_date):
        return DunningEligibility(
            eligible=False,
            reason="not_overdue_or_terminal",
            days_past_due=0,
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
            reason="collection_already_prepared",
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


def render_dunning_message(
    *,
    receivable: PortalReceivable,
    company: Company,
    event_type: str,
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
    body = (
        f"{title}\n"
        f"\n"
        f"Émetteur (créancier) : {creditor}\n"
        f"Destinataire (débiteur) : {debtor}\n"
        f"\n"
        f"Facture n° {invoice_no}\n"
        f"Date de facture : {_format_date(receivable.issued_at)}\n"
        f"Date d'échéance : {_format_date(receivable.due_date)}\n"
        f"Montant initial : {_format_chf(_money(receivable.total_amount))}\n"
        f"Paiements reçus :\n{paid_lines}\n"
        f"Solde restant dû : {_format_chf(_money(receivable.balance_due))}\n"
        f"\n"
        f"Merci de régulariser le solde auprès de {creditor}.\n"
        f"\n"
        f"—\n"
        f"Ce message est envoyé pour le compte de {creditor}.\n"
        f"Lirie fournit uniquement l'infrastructure technique.\n"
    )
    if event_type == DUNNING_FORMAL_NOTICE:
        body += (
            "\nSans règlement, le créancier pourra préparer un dossier de "
            "recouvrement. Aucune poursuite n'est déposée automatiquement.\n"
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
    """from_name = créancier ; reply_to = contact entreprise."""
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
) -> PortalReceivableDunningEvent:
    """Émet un événement. E-mail : succès requis pour delivery=sent."""
    company = db.session.get(Company, int(receivable.creditor_company_id))
    if company is None:
        raise PortalReceivableError(
            "Créancier introuvable.",
            code="creditor_not_found",
        )
    eligibility = resolve_dunning_eligibility(receivable, as_of=as_of)
    if (
        not force
        and (not eligibility.eligible or eligibility.next_event_type != event_type)
    ):
        raise PortalReceivableError(
            "Cette étape de rappel n'est pas encore due ou déjà effectuée.",
            code="dunning_not_due",
        )
    if event_type not in (
        DUNNING_REMINDER_1,
        DUNNING_REMINDER_2,
        DUNNING_FORMAL_NOTICE,
        DUNNING_COLLECTION_PREPARED,
    ):
        raise PortalReceivableError(
            "Type d'événement inconnu.",
            code="dunning_event_type_invalid",
        )

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
    )
    db.session.add(event)
    db.session.flush()
    return event


def serialize_dunning_event(event: PortalReceivableDunningEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "receivable_id": event.receivable_id,
        "creditor_company_id": event.creditor_company_id,
        "debtor_user_id": event.debtor_user_id,
        "event_type": event.event_type,
        "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
        "channel": event.channel,
        "recipient": event.recipient,
        "recipient_email": event.recipient_email,
        "template_version": event.template_version,
        "rendered_subject": event.rendered_subject,
        "rendered_body_hash": event.rendered_body_hash,
        "balance_due_snapshot": float(event.balance_due_snapshot),
        "due_date_snapshot": (
            event.due_date_snapshot.isoformat() if event.due_date_snapshot else None
        ),
        "external_invoice_number_snapshot": event.external_invoice_number_snapshot,
        "delivery_status": event.delivery_status,
        "provider_message_id": event.provider_message_id,
    }
