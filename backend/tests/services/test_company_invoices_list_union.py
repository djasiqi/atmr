"""Lot 6 — liste factures UNION ALL + pagination SQL."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from services.invoices.company_invoices_list import list_company_invoices_unified


def test_unified_list_empty_company(db, sample_company):
    items, total, stats = list_company_invoices_unified(
        company_id=sample_company.id, page=1, per_page=20
    )
    assert items == []
    assert total == 0
    assert stats["overdue_count"] == 0
    assert stats["total_issued"] == 0.0


def test_unified_list_orders_regular_by_issued_at(db, sample_company, sample_user):
    from models import Client, Invoice
    from models.enums import InvoiceStatus

    client = Client()
    client.company_id = sample_company.id
    client.user_id = sample_user.id
    db.session.add(client)
    db.session.flush()

    older = Invoice()
    older.company_id = sample_company.id
    older.client_id = client.id
    older.invoice_number = "INV-OLD"
    older.status = InvoiceStatus.SENT
    older.period_year = 2026
    older.period_month = 1
    older.total_amount = Decimal("10.00")
    older.amount_paid = Decimal("0")
    older.balance_due = Decimal("10.00")
    older.issued_at = datetime(2026, 1, 1, tzinfo=UTC)
    older.due_date = date(2026, 1, 31)

    newer = Invoice()
    newer.company_id = sample_company.id
    newer.client_id = client.id
    newer.invoice_number = "INV-NEW"
    newer.status = InvoiceStatus.SENT
    newer.period_year = 2026
    newer.period_month = 2
    newer.total_amount = Decimal("20.00")
    newer.amount_paid = Decimal("0")
    newer.balance_due = Decimal("20.00")
    newer.issued_at = datetime(2026, 2, 1, tzinfo=UTC)
    newer.due_date = date(2026, 2, 28)

    db.session.add_all([older, newer])
    db.session.flush()

    items, total, stats = list_company_invoices_unified(
        company_id=sample_company.id, page=1, per_page=10
    )
    assert total == 2
    assert items[0]["invoice_number"] == "INV-NEW"
    assert items[1]["invoice_number"] == "INV-OLD"
    assert stats["total_issued"] == 30.0

    page1, total1, _ = list_company_invoices_unified(
        company_id=sample_company.id, page=1, per_page=1
    )
    page2, total2, _ = list_company_invoices_unified(
        company_id=sample_company.id, page=2, per_page=1
    )
    assert total1 == total2 == 2
    assert len(page1) == 1
    assert page1[0]["invoice_number"] == "INV-NEW"
    assert len(page2) == 1
    assert page2[0]["invoice_number"] == "INV-OLD"
