"""Notifications cloche du portail client : messages du transporteur et factures reçues."""

from __future__ import annotations

from typing import Any


def serialize_client_message_notification(message: Any, booking: Any) -> dict[str, Any]:
    """Une réponse entreprise devient une ligne de cloche, sans le fil complet."""
    content = str(getattr(message, "content", "") or "").strip()
    if len(content) > 140:
        content = f"{content[:137]}…"
    sender = str(getattr(message, "sender_label", "") or "").strip() or "Transporteur"
    created = getattr(message, "created_at", None)
    created_iso = (
        created.isoformat() if created is not None and hasattr(created, "isoformat") else None
    )
    return {
        "id": getattr(message, "id", None),
        "event_type": "booking_message",
        "title": "Message du transporteur",
        "message": f"{sender} · {content}" if content else sender,
        "created_at": created_iso,
        "booking_id": getattr(booking, "id", None),
        "sender_label": sender,
    }


def serialize_client_invoice_notification(invoice: Any) -> dict[str, Any]:
    """Une facture émise au client devient une ligne de cloche."""
    number = str(getattr(invoice, "invoice_number", "") or "").strip()
    if not number:
        number = f"#{getattr(invoice, 'id', '')}"
    total = getattr(invoice, "total_amount", None)
    amount = ""
    try:
        if total is not None:
            amount = f"{float(total):.2f} CHF"
    except (TypeError, ValueError):
        amount = ""
    company = getattr(invoice, "company", None)
    company_name = str(getattr(company, "name", "") or "").strip()
    when = getattr(invoice, "sent_at", None) or getattr(invoice, "issued_at", None)
    created_iso = when.isoformat() if when is not None and hasattr(when, "isoformat") else None
    detail = " · ".join(part for part in (number, amount, company_name) if part)
    return {
        "id": f"invoice-{int(invoice.id)}",
        "event_type": "invoice_received",
        "title": "Facture reçue",
        "message": detail or "Une facture est disponible",
        "created_at": created_iso,
        "booking_id": None,
        "invoice_id": int(invoice.id),
    }
