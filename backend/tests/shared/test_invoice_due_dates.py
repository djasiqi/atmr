"""Échéances effectives facture / rappel."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from shared.invoice_due_dates import (
    compute_reminder_due_date,
    get_latest_open_reminder,
    get_reminder_payment_days_for_level,
    normalize_reminder_schedule_days,
    resolve_effective_due_date,
    resolve_reminder_due_date,
)


def _reminder(**kwargs):
    base = {
        "id": 1,
        "level": 2,
        "status": "OPEN",
        "generated_at": datetime(2026, 6, 17, tzinfo=UTC),
        "due_date": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _invoice(**kwargs):
    base = {
        "id": 10,
        "company_id": 1,
        "status": "sent",
        "balance_due": Decimal("55.50"),
        "due_date": datetime(2026, 6, 6, tzinfo=UTC),
        "reminders": [],
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_normalize_reminder_schedule_days_accepts_string_keys():
    schedule = normalize_reminder_schedule_days({"1": 15, "2": 10, "3": 5})
    assert schedule == {"1": 15, "2": 10, "3": 5}


def test_compute_reminder_due_date_adds_level_days():
    generated = datetime(2026, 6, 17, tzinfo=UTC)
    due = compute_reminder_due_date(generated, 10)
    assert due == datetime(2026, 6, 27, tzinfo=UTC)


def test_resolve_effective_due_date_uses_open_reminder():
    reminder = _reminder(due_date=datetime(2026, 6, 27, tzinfo=UTC))
    invoice = _invoice(reminders=[reminder])
    assert resolve_effective_due_date(invoice) == reminder.due_date


def test_resolve_effective_due_date_falls_back_to_invoice_when_no_open_reminder():
    reminder = _reminder(status="PAID", due_date=datetime(2026, 6, 27, tzinfo=UTC))
    invoice = _invoice(reminders=[reminder])
    assert resolve_effective_due_date(invoice) == invoice.due_date


def test_resolve_reminder_due_date_uses_level_override():
    reminder = _reminder(level=2, due_date=None)
    due = resolve_reminder_due_date(reminder, payment_days_override=10)
    assert due == datetime(2026, 6, 27, tzinfo=UTC)


def test_get_latest_open_reminder_prefers_highest_level():
    r1 = _reminder(id=1, level=1, generated_at=datetime(2026, 6, 1, tzinfo=UTC))
    r2 = _reminder(id=2, level=2, generated_at=datetime(2026, 6, 10, tzinfo=UTC))
    latest = get_latest_open_reminder(_invoice(reminders=[r1, r2]))
    assert latest is r2


def test_get_reminder_payment_days_for_level_defaults_without_company():
    assert get_reminder_payment_days_for_level(None, 2) == 5
