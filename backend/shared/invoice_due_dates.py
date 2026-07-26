"""Échéances effectives facture / rappel consolidé."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models import Invoice, InvoiceReminder

DEFAULT_REMINDER_SCHEDULE_DAYS: dict[int, int] = {1: 10, 2: 5, 3: 5}


def _schedule_key(level: int) -> str:
    return str(int(level))


def normalize_reminder_schedule_days(
    raw: dict[str, Any] | None,
) -> dict[str, int]:
    """Normalise le planning rappels (clés 1/2/3 en chaînes)."""
    if not raw:
        return {str(k): v for k, v in DEFAULT_REMINDER_SCHEDULE_DAYS.items()}
    normalized: dict[str, int] = {}
    for key in ("1", "2", "3"):
        value = raw.get(key, raw.get(int(key)))
        if value is not None:
            normalized[key] = max(int(value), 0)
    for level, default in DEFAULT_REMINDER_SCHEDULE_DAYS.items():
        normalized.setdefault(str(level), default)
    return normalized


def get_reminder_schedule_days(company_id: int | None) -> dict[str, int]:
    """Délais par niveau de rappel (jours accordés après émission du rappel)."""
    if not company_id:
        return {str(k): v for k, v in DEFAULT_REMINDER_SCHEDULE_DAYS.items()}
    from models import CompanyBillingSettings

    settings = CompanyBillingSettings.query.filter_by(company_id=company_id).first()
    if not settings:
        return {str(k): v for k, v in DEFAULT_REMINDER_SCHEDULE_DAYS.items()}
    return normalize_reminder_schedule_days(settings.reminder_schedule_days)


def get_reminder_payment_days_for_level(company_id: int | None, level: int) -> int:
    """Délai de paiement applicable à un niveau de rappel donné."""
    schedule = get_reminder_schedule_days(company_id)
    key = _schedule_key(level)
    return schedule.get(key, DEFAULT_REMINDER_SCHEDULE_DAYS.get(int(level), 10))


def compute_reminder_due_date(
    generated_at: datetime,
    payment_days: int,
) -> datetime:
    """Nouvelle échéance : date du rappel + délai du niveau."""
    base = generated_at if generated_at.tzinfo else generated_at.replace(tzinfo=UTC)
    return base + timedelta(days=max(int(payment_days), 0))


def resolve_reminder_due_date(
    reminder: InvoiceReminder,
    *,
    company_id: int | None = None,
    payment_days_override: int | None = None,
) -> datetime | None:
    """Échéance d'un rappel (colonne persistée ou calcul de repli)."""
    if reminder.due_date is not None:
        return reminder.due_date
    if not reminder.generated_at:
        return None
    days = payment_days_override
    if days is None:
        days = get_reminder_payment_days_for_level(company_id, int(reminder.level or 1))
    return compute_reminder_due_date(reminder.generated_at, days)


def get_latest_open_reminder(invoice: Invoice) -> InvoiceReminder | None:
    """Rappel OPEN le plus récent (niveau puis date de génération)."""
    reminders = getattr(invoice, "reminders", None) or []
    open_reminders = [r for r in reminders if (r.status or "OPEN") == "OPEN"]
    if not open_reminders:
        return None
    return max(
        open_reminders,
        key=lambda r: (
            r.level or 0,
            r.generated_at or datetime.min.replace(tzinfo=UTC),
        ),
    )


def resolve_effective_due_date(
    invoice: Invoice,
    *,
    company_id: int | None = None,
) -> datetime | None:
    """Échéance affichée : rappel OPEN en cours, sinon facture initiale."""
    invoice_due_date = getattr(invoice, "due_date", None)
    if invoice_due_date is None:
        return None

    status = (
        invoice.status.value
        if hasattr(invoice.status, "value")
        else str(invoice.status)
    )
    if status in ("paid", "cancelled", "draft"):
        return invoice_due_date
    if float(getattr(invoice, "balance_due", 0) or 0) <= 0:
        return invoice_due_date

    latest_open = get_latest_open_reminder(invoice)
    if latest_open is None:
        return invoice_due_date

    resolved_company_id = company_id or getattr(invoice, "company_id", None)
    return (
        resolve_reminder_due_date(latest_open, company_id=resolved_company_id)
        or invoice_due_date
    )
