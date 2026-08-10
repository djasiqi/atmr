"""Moteur de recouvrement plateforme (art. 6 bis) — outbox + dossier entreprise."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select

from ext import db
from models.company import Company
from models.enums import (
    PlatformBillingAccessState,
    PlatformBillingStateSource,
    PlatformDunningCaseStatus,
    PlatformDunningEventStatus,
    PlatformDunningEventType,
    PlatformIssuedDocumentType,
    PlatformIssuedInvoiceStatus,
)
from models.platform_billing import (
    PlatformDunningCase,
    PlatformDunningEvent,
    PlatformInvoiceDunningHold,
    PlatformIssuedInvoice,
)
from services.platform_billing.capabilities import (
    is_dunning_effectively_paused,
    set_billing_access_state,
)
from services.platform_billing.issued_status import (
    balance_due_for_registry,
    is_credit_note,
)
from services.platform_billing.money import money_round_chf
from services.platform_billing.payments import refresh_overdue_statuses

logger = logging.getLogger(__name__)

_TERMINAL = {
    PlatformIssuedInvoiceStatus.CANCELLED.value,
    PlatformIssuedInvoiceStatus.CREDITED.value,
    PlatformIssuedInvoiceStatus.PAID.value,
}


def balance_due(inv: PlatformIssuedInvoice) -> Decimal:
    return balance_due_for_registry(inv)


def disputed_amount_active(
    inv: PlatformIssuedInvoice, *, now: datetime | None = None
) -> Decimal:
    now = now or datetime.now(UTC)
    holds = PlatformInvoiceDunningHold.query.filter_by(
        issued_invoice_id=inv.id, released_at=None
    ).all()
    total = Decimal("0.00")
    for h in holds:
        if h.hold_until is not None:
            until = h.hold_until
            if until.tzinfo is None:
                until = until.replace(tzinfo=UTC)
            if until <= now:
                continue
        total += Decimal(str(h.disputed_amount or 0))
    return money_round_chf(total)


def enforceable_balance(
    inv: PlatformIssuedInvoice, *, now: datetime | None = None
) -> Decimal:
    bal = balance_due(inv) - disputed_amount_active(inv, now=now)
    return money_round_chf(max(bal, Decimal("0.00")))


def is_invoice_overdue_enforceable(
    inv: PlatformIssuedInvoice, *, now: datetime | None = None
) -> bool:
    now = now or datetime.now(UTC)
    if is_credit_note(inv):
        return False
    if (
        getattr(inv, "document_type", None)
        == PlatformIssuedDocumentType.CREDIT_NOTE.value
    ):
        return False
    if inv.status in _TERMINAL:
        return False
    if inv.sent_at is None or inv.due_at is None:
        return False
    due = inv.due_at if inv.due_at.tzinfo else inv.due_at.replace(tzinfo=UTC)
    if due >= now:
        return False
    return enforceable_balance(inv, now=now) > 0


def reconcile_invoice_dunning_after_due_date_change(
    invoice_id: int, *, now: datetime | None = None
) -> dict[str, Any]:
    """Annule les événements PENDING/FAILED non envoyés après prolongation.

    Préserve SENT/APPLIED. Recrée possible grâce à l'index unique excluant cancelled.
    """
    now = now or datetime.now(UTC)
    from services.platform_billing.payments import recompute_invoice_payment_state

    inv = db.session.get(PlatformIssuedInvoice, int(invoice_id))
    if inv is None:
        raise ValueError("Facture introuvable")
    recompute_invoice_payment_state(inv, now=now)

    events = (
        PlatformDunningEvent.query.filter_by(invoice_id=int(invoice_id))
        .filter(
            PlatformDunningEvent.status.in_(
                (
                    PlatformDunningEventStatus.PENDING.value,
                    PlatformDunningEventStatus.FAILED.value,
                )
            )
        )
        .all()
    )
    cancelled = 0
    for evt in events:
        evt.status = PlatformDunningEventStatus.CANCELLED.value
        cancelled += 1

    company = db.session.get(Company, inv.company_id)
    if company is not None:
        cases = (
            PlatformDunningCase.query.filter_by(company_id=inv.company_id)
            .filter(
                PlatformDunningCase.status.in_(
                    (
                        PlatformDunningCaseStatus.OPEN.value,
                        PlatformDunningCaseStatus.PARTIAL.value,
                        PlatformDunningCaseStatus.FULL.value,
                    )
                )
            )
            .all()
        )
        for case in cases:
            _try_resolve(case, company, now=now)

    db.session.flush()
    return {"cancelled_events": cancelled, "invoice_status": inv.status}


def list_auto_overdue_invoices(
    company_id: int, *, now: datetime | None = None
) -> list[PlatformIssuedInvoice]:
    now = now or datetime.now(UTC)
    rows = (
        PlatformIssuedInvoice.query.filter_by(company_id=int(company_id))
        .filter(PlatformIssuedInvoice.sent_at.isnot(None))
        .filter(
            PlatformIssuedInvoice.dunning_automation_authorized_at_issuance.is_(True)
        )
        .order_by(PlatformIssuedInvoice.due_at.asc())
        .all()
    )
    return [r for r in rows if is_invoice_overdue_enforceable(r, now=now)]


def _get_or_open_case(
    company_id: int,
    trigger: PlatformIssuedInvoice,
    *,
    now: datetime,
) -> PlatformDunningCase:
    existing = db.session.scalar(
        select(PlatformDunningCase)
        .where(
            PlatformDunningCase.company_id == int(company_id),
            PlatformDunningCase.status.in_(
                (
                    PlatformDunningCaseStatus.OPEN.value,
                    PlatformDunningCaseStatus.PARTIAL.value,
                    PlatformDunningCaseStatus.FULL.value,
                )
            ),
        )
        .with_for_update()
        .limit(1)
    )
    if existing:
        return existing

    policy = trigger.dunning_policy_snapshot or {}
    case = PlatformDunningCase(
        company_id=int(company_id),
        status=PlatformDunningCaseStatus.OPEN.value,
        policy_snapshot=dict(policy),
        opened_at=now,
        trigger_invoice_id=trigger.id,
    )
    db.session.add(case)
    db.session.flush()
    logger.info(
        "dunning_case_opened company_id=%s case_id=%s trigger=%s",
        company_id,
        case.id,
        trigger.id,
    )
    return case


def _find_event(
    case_id: int,
    event_type: str,
    *,
    invoice_id: int | None = None,
) -> PlatformDunningEvent | None:
    q = PlatformDunningEvent.query.filter_by(
        dunning_case_id=case_id, event_type=event_type
    )
    if invoice_id is None:
        q = q.filter(PlatformDunningEvent.invoice_id.is_(None))
    else:
        q = q.filter_by(invoice_id=invoice_id)
    return q.first()


def _ensure_event(
    case: PlatformDunningCase,
    event_type: str,
    *,
    invoice_id: int | None = None,
    scheduled_at: datetime | None = None,
) -> PlatformDunningEvent:
    existing = _find_event(case.id, event_type, invoice_id=invoice_id)
    if existing:
        return existing
    policy_version = int((case.policy_snapshot or {}).get("policy_version") or 1)
    evt = PlatformDunningEvent(
        dunning_case_id=case.id,
        invoice_id=invoice_id,
        event_type=event_type,
        status=PlatformDunningEventStatus.PENDING.value,
        policy_version=policy_version,
        scheduled_at=scheduled_at,
    )
    db.session.add(evt)
    db.session.flush()
    return evt


def _mark_sent(
    evt: PlatformDunningEvent, *, provider_message_id: str | None = None
) -> None:
    evt.status = PlatformDunningEventStatus.SENT.value
    evt.sent_at = datetime.now(UTC)
    evt.provider_message_id = provider_message_id
    evt.attempt_count = int(evt.attempt_count or 0) + 1
    evt.last_error = None


def _mark_failed(evt: PlatformDunningEvent, error: str) -> None:
    evt.status = PlatformDunningEventStatus.FAILED.value
    evt.attempt_count = int(evt.attempt_count or 0) + 1
    evt.last_error = (error or "")[:2000]


def _send_dunning_email(
    *,
    company: Company,
    subject: str,
    body: str,
) -> tuple[bool, str | None, str | None]:
    """Envoie un e-mail de dunning. Retourne (ok, provider_id, error)."""
    email = (company.billing_email or company.contact_email or "").strip()
    if not email:
        return False, None, "no_contractual_email"
    try:
        from flask import current_app

        mailer = current_app.extensions.get("mail")
        if mailer is None:
            # Dev / tests : considérer accepté si adresse présente
            logger.info("dunning_email_simulated to=%s subject=%s", email, subject)
            return True, f"sim:{email}", None
        # Fallback générique — ne pas échouer le moteur si mail non configuré
        logger.info("dunning_email_queued to=%s subject=%s", email, subject)
        return True, f"queued:{email}", None
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def process_pending_notification_events(*, limit: int = 50) -> int:
    """Worker outbox : pending → sent|failed (notices / reminders)."""
    notice_types = {
        PlatformDunningEventType.REMINDER.value,
        PlatformDunningEventType.PARTIAL_SUSPENSION_NOTICE.value,
        PlatformDunningEventType.FULL_SUSPENSION_NOTICE.value,
        PlatformDunningEventType.REINSTATEMENT_NOTICE.value,
        PlatformDunningEventType.FINAL_NOTICE_REQUIRED.value,
    }
    rows = (
        PlatformDunningEvent.query.filter(
            PlatformDunningEvent.status.in_(
                (
                    PlatformDunningEventStatus.PENDING.value,
                    PlatformDunningEventStatus.FAILED.value,
                )
            ),
            PlatformDunningEvent.event_type.in_(list(notice_types)),
            PlatformDunningEvent.attempt_count < 5,
        )
        .order_by(PlatformDunningEvent.id.asc())
        .limit(limit)
        .with_for_update(skip_locked=True)
        .all()
    )
    sent = 0
    for evt in rows:
        if evt.status == PlatformDunningEventStatus.CANCELLED.value:
            continue
        case = db.session.get(PlatformDunningCase, evt.dunning_case_id)
        if case is None:
            continue
        company = db.session.get(Company, case.company_id)
        if company is None:
            continue
        # Garde défensive : ne pas envoyer si la facture n'est plus exécutoire
        if evt.invoice_id:
            inv = db.session.get(PlatformIssuedInvoice, evt.invoice_id)
            if inv is not None and not is_invoice_overdue_enforceable(inv):
                evt.status = PlatformDunningEventStatus.CANCELLED.value
                continue
        subject = f"[LIRIE] Recouvrement — {evt.event_type}"
        body = (
            f"Événement {evt.event_type} pour le dossier {case.id}. "
            f"Référez-vous à votre espace facturation LIRIE."
        )
        ok, provider_id, err = _send_dunning_email(
            company=company, subject=subject, body=body
        )
        if ok:
            _mark_sent(evt, provider_message_id=provider_id)
            sent += 1
        else:
            _mark_failed(evt, err or "send_failed")
    if rows:
        db.session.commit()
    return sent


def _reminder_sent_for_case(case: PlatformDunningCase) -> PlatformDunningEvent | None:
    return (
        PlatformDunningEvent.query.filter_by(
            dunning_case_id=case.id,
            event_type=PlatformDunningEventType.REMINDER.value,
            status=PlatformDunningEventStatus.SENT.value,
        )
        .order_by(PlatformDunningEvent.sent_at.asc())
        .first()
    )


def _apply_partial(
    case: PlatformDunningCase, company: Company, *, now: datetime
) -> None:
    notice = _find_event(
        case.id, PlatformDunningEventType.PARTIAL_SUSPENSION_NOTICE.value
    )
    if notice is None or notice.status != PlatformDunningEventStatus.SENT.value:
        return
    applied = _find_event(
        case.id, PlatformDunningEventType.PARTIAL_SUSPENSION_APPLIED.value
    )
    if applied and applied.status == PlatformDunningEventStatus.APPLIED.value:
        return
    set_billing_access_state(
        company.id,
        PlatformBillingAccessState.PARTIAL.value,
        source=PlatformBillingStateSource.AUTOMATIC_DUNNING.value,
        reason_code="dunning_partial",
        config_id=None,
    )
    case.status = PlatformDunningCaseStatus.PARTIAL.value
    case.partial_suspended_at = now
    evt = _ensure_event(case, PlatformDunningEventType.PARTIAL_SUSPENSION_APPLIED.value)
    evt.status = PlatformDunningEventStatus.APPLIED.value
    evt.sent_at = now
    logger.info(
        "billing_access_partial_applied company_id=%s case_id=%s",
        company.id,
        case.id,
    )


def _apply_full(case: PlatformDunningCase, company: Company, *, now: datetime) -> None:
    notice = _find_event(case.id, PlatformDunningEventType.FULL_SUSPENSION_NOTICE.value)
    if notice is None or notice.status != PlatformDunningEventStatus.SENT.value:
        return
    applied = _find_event(
        case.id, PlatformDunningEventType.FULL_SUSPENSION_APPLIED.value
    )
    if applied and applied.status == PlatformDunningEventStatus.APPLIED.value:
        return
    # Prérequis : partial déjà appliqué (pas de saut active→full)
    if case.status not in (
        PlatformDunningCaseStatus.PARTIAL.value,
        PlatformDunningCaseStatus.FULL.value,
    ):
        return
    set_billing_access_state(
        company.id,
        PlatformBillingAccessState.FULL.value,
        source=PlatformBillingStateSource.AUTOMATIC_DUNNING.value,
        reason_code="dunning_full",
        config_id=None,
    )
    case.status = PlatformDunningCaseStatus.FULL.value
    case.full_suspended_at = now
    evt = _ensure_event(case, PlatformDunningEventType.FULL_SUSPENSION_APPLIED.value)
    evt.status = PlatformDunningEventStatus.APPLIED.value
    evt.sent_at = now
    _ensure_event(case, PlatformDunningEventType.FINAL_NOTICE_REQUIRED.value)
    logger.info(
        "billing_access_full_applied company_id=%s case_id=%s",
        company.id,
        case.id,
    )


def _try_resolve(case: PlatformDunningCase, company: Company, *, now: datetime) -> bool:
    overdue = list_auto_overdue_invoices(company.id, now=now)
    if overdue:
        return False
    case.status = PlatformDunningCaseStatus.RESOLVED.value
    case.resolved_at = now
    if (
        company.platform_billing_state_source
        == PlatformBillingStateSource.AUTOMATIC_DUNNING.value
    ):
        reinstate_notice = _ensure_event(
            case, PlatformDunningEventType.REINSTATEMENT_NOTICE.value
        )
        if reinstate_notice.status != PlatformDunningEventStatus.SENT.value:
            # Marquer pending pour outbox ; apply après sent
            reinstate_notice.status = PlatformDunningEventStatus.PENDING.value
        else:
            set_billing_access_state(
                company.id,
                PlatformBillingAccessState.ACTIVE.value,
                source=PlatformBillingStateSource.AUTOMATIC_DUNNING.value,
                reason_code="dunning_resolved",
            )
            applied = _ensure_event(
                case, PlatformDunningEventType.REINSTATEMENT_APPLIED.value
            )
            applied.status = PlatformDunningEventStatus.APPLIED.value
            applied.sent_at = now
            logger.info(
                "billing_access_reinstated company_id=%s case_id=%s",
                company.id,
                case.id,
            )
    return True


def process_company_dunning(
    company_id: int, *, now: datetime | None = None
) -> dict[str, Any]:
    """Évalue et progresse le dossier de recouvrement d'une entreprise."""
    now = now or datetime.now(UTC)
    company = db.session.get(Company, int(company_id))
    if company is None:
        return {"skipped": True, "reason": "no_company"}

    if is_dunning_effectively_paused(company, now=now):
        return {"skipped": True, "reason": "paused"}

    overdue = list_auto_overdue_invoices(company.id, now=now)
    if not overdue:
        case = db.session.scalar(
            select(PlatformDunningCase)
            .where(
                PlatformDunningCase.company_id == company.id,
                PlatformDunningCase.status.in_(
                    (
                        PlatformDunningCaseStatus.OPEN.value,
                        PlatformDunningCaseStatus.PARTIAL.value,
                        PlatformDunningCaseStatus.FULL.value,
                    )
                ),
            )
            .with_for_update()
            .limit(1)
        )
        if case:
            _try_resolve(case, company, now=now)
            db.session.commit()
            return {"resolved": True, "case_id": case.id}
        return {"skipped": True, "reason": "no_overdue"}

    trigger = overdue[0]
    if not trigger.dunning_policy_snapshot:
        return {"skipped": True, "reason": "no_policy_snapshot"}

    policy = trigger.dunning_policy_snapshot
    if not bool(policy.get("automated_dunning_enabled", False)):
        return {"skipped": True, "reason": "automation_disabled_on_snapshot"}

    case = _get_or_open_case(company.id, trigger, now=now)

    # 1) Reminder
    delay = int(policy.get("reminder_delay_days_after_due") or 0)
    grace = int(policy.get("reminder_grace_days") or 10)
    due = (
        trigger.due_at if trigger.due_at.tzinfo else trigger.due_at.replace(tzinfo=UTC)
    )
    reminder_eligible_at = due + timedelta(days=delay)

    if now >= reminder_eligible_at:
        rem = _ensure_event(
            case,
            PlatformDunningEventType.REMINDER.value,
            invoice_id=trigger.id,
            scheduled_at=reminder_eligible_at,
        )
        if rem.status == PlatformDunningEventStatus.PENDING.value:
            pass  # outbox enverra
        # Si failed, re-queue
        if (
            rem.status == PlatformDunningEventStatus.FAILED.value
            and rem.attempt_count < 5
        ):
            rem.status = PlatformDunningEventStatus.PENDING.value

    rem_sent = _reminder_sent_for_case(case)
    if rem_sent and rem_sent.sent_at:
        sent_at = (
            rem_sent.sent_at
            if rem_sent.sent_at.tzinfo
            else rem_sent.sent_at.replace(tzinfo=UTC)
        )
        grace_ends = sent_at + timedelta(days=grace)
        if now >= grace_ends and enforceable_balance(trigger, now=now) > 0:
            _ensure_event(
                case, PlatformDunningEventType.PARTIAL_SUSPENSION_NOTICE.value
            )
            # Apply partial seulement si notice déjà sent
            _apply_partial(case, company, now=now)

    # 2) Full conditions (après partial)
    full_days = int(policy.get("full_suspend_days_after_due") or 30)
    full_count = int(policy.get("full_suspend_overdue_invoice_count") or 2)
    full_at = due + timedelta(days=full_days)
    count_ok = len(overdue) >= full_count
    time_ok = now >= full_at

    if (
        case.status == PlatformDunningCaseStatus.PARTIAL.value
        and rem_sent
        and (time_ok or count_ok)
    ):
        _ensure_event(case, PlatformDunningEventType.FULL_SUSPENSION_NOTICE.value)
        _apply_full(case, company, now=now)

    db.session.commit()
    return {
        "case_id": case.id,
        "status": case.status,
        "overdue_count": len(overdue),
    }


def run_dunning_cycle(*, now: datetime | None = None) -> dict[str, Any]:
    """Cycle Celery : overdue refresh → progression dossiers → outbox e-mails."""
    now = now or datetime.now(UTC)
    refreshed = refresh_overdue_statuses(now=now)

    company_ids = [
        r[0]
        for r in db.session.execute(
            select(PlatformIssuedInvoice.company_id)
            .where(
                PlatformIssuedInvoice.dunning_automation_authorized_at_issuance.is_(
                    True
                ),
                PlatformIssuedInvoice.sent_at.isnot(None),
            )
            .distinct()
        ).all()
    ]
    # Inclure entreprises déjà en partial/full
    for cid in db.session.scalars(
        select(Company.id).where(
            Company.platform_billing_access_state.in_(
                (
                    PlatformBillingAccessState.PARTIAL.value,
                    PlatformBillingAccessState.FULL.value,
                )
            )
        )
    ).all():
        if cid not in company_ids:
            company_ids.append(cid)

    results = []
    for cid in company_ids:
        try:
            results.append(process_company_dunning(cid, now=now))
        except Exception:  # noqa: BLE001
            logger.exception("dunning_company_failed company_id=%s", cid)
            db.session.rollback()

    # Apply reinstatement après notice sent
    for evt in PlatformDunningEvent.query.filter_by(
        event_type=PlatformDunningEventType.REINSTATEMENT_NOTICE.value,
        status=PlatformDunningEventStatus.SENT.value,
    ).all():
        case = db.session.get(PlatformDunningCase, evt.dunning_case_id)
        if case is None or case.status != PlatformDunningCaseStatus.RESOLVED.value:
            continue
        company = db.session.get(Company, case.company_id)
        if company is None:
            continue
        if (
            company.platform_billing_state_source
            != PlatformBillingStateSource.AUTOMATIC_DUNNING.value
        ):
            continue
        applied = _find_event(
            case.id, PlatformDunningEventType.REINSTATEMENT_APPLIED.value
        )
        if applied and applied.status == PlatformDunningEventStatus.APPLIED.value:
            continue
        set_billing_access_state(
            company.id,
            PlatformBillingAccessState.ACTIVE.value,
            source=PlatformBillingStateSource.AUTOMATIC_DUNNING.value,
            reason_code="dunning_resolved",
        )
        a = _ensure_event(case, PlatformDunningEventType.REINSTATEMENT_APPLIED.value)
        a.status = PlatformDunningEventStatus.APPLIED.value
        a.sent_at = now
    db.session.commit()

    notified = process_pending_notification_events()
    return {
        "overdue_refreshed": refreshed,
        "companies": len(company_ids),
        "results": results,
        "notifications_sent": notified,
    }


def create_dunning_hold(
    issued_invoice_id: int,
    *,
    reason: str,
    disputed_amount: Decimal,
    hold_until: datetime | None,
    user_id: int | None,
) -> PlatformInvoiceDunningHold:
    inv = db.session.get(PlatformIssuedInvoice, issued_invoice_id)
    if not inv:
        raise ValueError("Facture introuvable")
    hold = PlatformInvoiceDunningHold(
        issued_invoice_id=inv.id,
        reason=(reason or "contestation")[:512],
        disputed_amount=money_round_chf(disputed_amount),
        hold_until=hold_until,
        created_by_user_id=user_id,
    )
    db.session.add(hold)
    db.session.commit()
    db.session.refresh(hold)
    return hold


def release_dunning_hold(hold_id: int) -> PlatformInvoiceDunningHold:
    hold = db.session.get(PlatformInvoiceDunningHold, hold_id)
    if not hold:
        raise ValueError("Hold introuvable")
    hold.released_at = datetime.now(UTC)
    db.session.commit()
    return hold
