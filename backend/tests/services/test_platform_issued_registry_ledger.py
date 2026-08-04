"""Tests ledger / registres factures plateforme (PR1) — unitaires sans DB."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from models.enums import (
    PlatformIssuedDocumentType,
    PlatformIssuedInvoiceStatus,
)
from services.platform_billing.issued_status import (
    balance_due_for_registry,
    payment_state,
    ui_status,
)
from services.platform_billing.payments import (
    create_credit_note,
    recompute_invoice_payment_state,
)


def _make_invoice(**kwargs):
    defaults = dict(
        id=1,
        company_id=1,
        invoice_number="LIRIE-2026-08-0001",
        document_type=PlatformIssuedDocumentType.INVOICE.value,
        status=PlatformIssuedInvoiceStatus.SENT.value,
        currency="CHF",
        subtotal_amount=Decimal("100.00"),
        tax_rate=Decimal("0"),
        tax_amount=Decimal("0.00"),
        total_amount=Decimal("100.00"),
        qr_amount=Decimal("100.00"),
        amount_paid=Decimal("0.00"),
        issued_at=datetime(2026, 8, 1, tzinfo=UTC),
        due_at=datetime(2026, 8, 31, tzinfo=UTC),
        sent_at=datetime(2026, 8, 2, tzinfo=UTC),
        paid_at=None,
        billing_year=2026,
        billing_month=8,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestIssuedStatus:
    def test_partial_overdue_ui_priority(self):
        inv = _make_invoice(
            amount_paid=Decimal("40.00"),
            due_at=datetime(2026, 7, 1, tzinfo=UTC),
            status=PlatformIssuedInvoiceStatus.OVERDUE.value,
        )
        now = datetime(2026, 8, 4, tzinfo=UTC)
        assert payment_state(inv) == "PARTIAL"
        assert ui_status(inv, now=now) == "OVERDUE"

    def test_credit_note_balance_zero(self):
        inv = _make_invoice(
            document_type=PlatformIssuedDocumentType.CREDIT_NOTE.value,
            total_amount=Decimal("-100.00"),
            status=PlatformIssuedInvoiceStatus.ISSUED.value,
            sent_at=None,
        )
        assert balance_due_for_registry(inv) == Decimal("0.00")
        assert payment_state(inv) == "NONE"
        assert ui_status(inv) == "ISSUED"

    def test_partial_not_overdue(self):
        inv = _make_invoice(
            amount_paid=Decimal("40.00"),
            due_at=datetime(2026, 12, 31, tzinfo=UTC),
            status=PlatformIssuedInvoiceStatus.SENT.value,
        )
        now = datetime(2026, 8, 4, tzinfo=UTC)
        assert ui_status(inv, now=now) == "PARTIALLY_PAID"


def test_recompute_clears_paid_at_after_partial_reverse_logic():
    inv = _make_invoice(
        amount_paid=Decimal("100.00"),
        status=PlatformIssuedInvoiceStatus.PAID.value,
        paid_at=datetime(2026, 8, 3, tzinfo=UTC),
        sent_at=datetime(2026, 8, 2, tzinfo=UTC),
        due_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    with patch(
        "services.platform_billing.payments.sum_ledger_amount",
        return_value=Decimal("40.00"),
    ):
        recompute_invoice_payment_state(
            inv, now=datetime(2026, 8, 4, tzinfo=UTC)
        )
    assert inv.amount_paid == Decimal("40.00")
    assert inv.paid_at is None
    assert inv.status == PlatformIssuedInvoiceStatus.SENT.value


def test_recompute_overdue_after_reverse():
    inv = _make_invoice(
        amount_paid=Decimal("100.00"),
        status=PlatformIssuedInvoiceStatus.PAID.value,
        paid_at=datetime(2026, 8, 3, tzinfo=UTC),
        due_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    with patch(
        "services.platform_billing.payments.sum_ledger_amount",
        return_value=Decimal("40.00"),
    ):
        recompute_invoice_payment_state(
            inv, now=datetime(2026, 8, 4, tzinfo=UTC)
        )
    assert inv.status == PlatformIssuedInvoiceStatus.OVERDUE.value
    assert inv.paid_at is None


def test_credit_note_rejects_when_paid():
    inv = _make_invoice(amount_paid=Decimal("10.00"))
    with patch(
        "services.platform_billing.payments._lock_invoice", return_value=inv
    ):
        with pytest.raises(ValueError, match="paiement"):
            create_credit_note(1, reason="Erreur de facturation")


def test_credit_note_requires_reason():
    inv = _make_invoice(amount_paid=Decimal("0.00"))
    with patch(
        "services.platform_billing.payments._lock_invoice", return_value=inv
    ):
        with pytest.raises(ValueError, match="Motif"):
            create_credit_note(1, reason="  ")
