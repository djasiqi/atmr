"""Sérialisation liste factures : rappels allégés pour accès PDF."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from models.invoice import Invoice, InvoiceReminder


def _minimal_invoice(**overrides):
    """Facture minimale pour tester to_dict(list_view=True) sans DB."""
    base = {
        "id": 10,
        "company_id": 1,
        "client_id": 2,
        "bill_to_client_id": None,
        "billing_party_id": None,
        "billing_strategy": "legacy",
        "billed_to_company_id": None,
        "period_month": 5,
        "period_year": 2026,
        "invoice_number": "EM-2026-05-0001",
        "currency": "CHF",
        "subtotal_amount": Decimal("55.50"),
        "late_fee_amount": Decimal("0"),
        "reminder_fee_amount": Decimal("0"),
        "vat_total_amount": Decimal("0"),
        "vat_breakdown": None,
        "total_amount": Decimal("55.50"),
        "amount_paid": Decimal("0"),
        "balance_due": Decimal("55.50"),
        "issued_at": datetime(2026, 5, 1, tzinfo=UTC),
        "due_date": datetime(2026, 6, 6, tzinfo=UTC),
        "sent_at": None,
        "paid_at": None,
        "cancelled_at": None,
        "created_at": datetime(2026, 5, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 5, 1, tzinfo=UTC),
        "status": "sent",
        "reminder_level": 2,
        "last_reminder_at": datetime(2026, 6, 10, tzinfo=UTC),
        "pdf_url": "/uploads/invoices/inv.pdf",
        "qr_reference": None,
        "meta": None,
        "client": None,
        "bill_to_client": None,
        "billing_party": None,
        "billed_to_company": None,
        "lines": [],
        "payments": [],
        "reminders": [],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_reminder_to_dict_list_view_exposes_pdf_url():
    reminder = InvoiceReminder()
    reminder.id = 42
    reminder.level = 2
    reminder.pdf_url = "/uploads/reminders/reminder_2.pdf"
    reminder.status = "OPEN"
    reminder.generated_at = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    reminder.reminder_fee_amount = Decimal("40.00")

    payload = reminder.to_dict_list_view()

    assert payload["id"] == 42
    assert payload["level"] == 2
    assert payload["pdf_url"] == "/uploads/reminders/reminder_2.pdf"
    assert payload["status"] == "OPEN"
    assert payload["reminder_fee_amount"] == 40.0
    assert "principal_amount" not in payload


def test_invoice_to_dict_list_view_includes_lightweight_reminders():
    reminder = InvoiceReminder()
    reminder.id = 7
    reminder.level = 2
    reminder.pdf_url = "/uploads/reminders/r2.pdf"
    reminder.status = "OPEN"
    reminder.generated_at = datetime(2026, 6, 10, tzinfo=UTC)
    reminder.reminder_fee_amount = Decimal("40")
    reminder.due_date = datetime(2026, 6, 20, tzinfo=UTC)

    invoice_data = _minimal_invoice(reminders=[reminder])
    invoice = Invoice()
    for key, value in invoice_data.__dict__.items():
        setattr(invoice, key, value)

    with patch.object(Invoice, "_serialize_client", return_value=None):
        payload = invoice.to_dict(list_view=True, company_id=1)

    assert payload["lines"] == []
    assert payload["payments"] == []
    assert len(payload["reminders"]) == 1
    assert payload["reminders"][0]["pdf_url"] == "/uploads/reminders/r2.pdf"
    assert payload["reminders"][0]["level"] == 2
    assert payload["reminders"][0]["due_date"] is not None
    assert payload["effective_due_date"] is not None
    assert "principal_amount" not in payload["reminders"][0]
