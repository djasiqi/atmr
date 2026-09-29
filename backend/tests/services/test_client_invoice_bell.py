"""Cloche client : une facture émise devient une notification."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from services.notifications.client_message_bell import serialize_client_invoice_notification


def test_invoice_notification_names_the_received_bill():
    invoice = SimpleNamespace(
        id=2318,
        invoice_number="EM-2026-09-0010",
        total_amount=Decimal("80.00"),
        sent_at=datetime(2026, 9, 29, 18, 0, tzinfo=UTC),
        issued_at=datetime(2026, 9, 29, 17, 0, tzinfo=UTC),
        company=SimpleNamespace(name="Emmenez-moi"),
    )
    payload = serialize_client_invoice_notification(invoice)
    assert payload["id"] == "invoice-2318"
    assert payload["event_type"] == "invoice_received"
    assert payload["title"] == "Facture reçue"
    assert payload["invoice_id"] == 2318
    assert "EM-2026-09-0010" in payload["message"]
    assert "80.00 CHF" in payload["message"]
    assert "Emmenez-moi" in payload["message"]
    assert payload["created_at"].startswith("2026-09-29T18:00")
