#!/usr/bin/env python
"""Consultation facture + paiements / rappels (ops / support).

Utilise l'ORM — schéma réel :
- table ``client`` (pas ``clients``) ; prénom/nom sur ``user``
- ``invoice_payments`` : id, invoice_id, amount, paid_at, method, reference, reminder_id
  (pas de colonne ``created_at``)

Usage:
    python scripts/lookup_invoice.py EM-2026-02-0034
    python scripts/lookup_invoice.py --id 373
    python scripts/lookup_invoice.py --json EM-2026-02-0034
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from sqlalchemy.orm import joinedload

from app import create_app
from models import Client, Invoice, InvoicePayment, InvoiceReminder


def _client_name(client: Client | None) -> str | None:
    if client is None:
        return None
    user = getattr(client, "user", None)
    if user is None:
        return None
    parts = [getattr(user, "first_name", "") or "", getattr(user, "last_name", "") or ""]
    name = " ".join(p for p in parts if p).strip()
    return name or getattr(user, "username", None)


def _serialize_payment(payment: InvoicePayment) -> dict:
    method = payment.method.value if hasattr(payment.method, "value") else str(payment.method)
    return {
        "id": payment.id,
        "invoice_id": payment.invoice_id,
        "amount": float(payment.amount or 0),
        "paid_at": payment.paid_at.isoformat() if payment.paid_at else None,
        "method": method,
        "reference": payment.reference,
        "reminder_id": payment.reminder_id,
    }


def _serialize_reminder(reminder: InvoiceReminder) -> dict:
    return {
        "id": reminder.id,
        "invoice_id": reminder.invoice_id,
        "level": reminder.level,
        "added_fee": float(reminder.added_fee or 0),
        "principal_amount": float(reminder.principal_amount or 0),
        "reminder_fee_amount": float(reminder.reminder_fee_amount or 0),
        "total_due": float(reminder.total_due or 0),
        "sent_at": reminder.sent_at.isoformat() if reminder.sent_at else None,
        "due_date": reminder.due_date.isoformat() if reminder.due_date else None,
    }


def lookup_invoice(
    *,
    invoice_number: str | None = None,
    invoice_id: int | None = None,
) -> dict | None:
    query = Invoice.query.options(
        joinedload(Invoice.client).joinedload(Client.user),
        joinedload(Invoice.bill_to_client).joinedload(Client.user),
        joinedload(Invoice.payments),
        joinedload(Invoice.reminders),
    )
    if invoice_id is not None:
        invoice = query.filter(Invoice.id == int(invoice_id)).first()
    elif invoice_number:
        invoice = query.filter(Invoice.invoice_number == invoice_number.strip()).first()
    else:
        return None

    if invoice is None:
        return None

    status = invoice.status.value if hasattr(invoice.status, "value") else str(invoice.status)
    payments = sorted(invoice.payments or [], key=lambda p: p.paid_at or "")
    reminders = sorted(invoice.reminders or [], key=lambda r: r.level or 0)

    return {
        "id": invoice.id,
        "invoice_number": invoice.invoice_number,
        "company_id": invoice.company_id,
        "client_id": invoice.client_id,
        "client_name": _client_name(invoice.client),
        "bill_to_client_id": invoice.bill_to_client_id,
        "bill_to_client_name": _client_name(invoice.bill_to_client),
        "status": status,
        "total_amount": float(invoice.total_amount or 0),
        "amount_paid": float(invoice.amount_paid or 0),
        "balance_due": float(invoice.balance_due or 0),
        "issued_at": invoice.issued_at.isoformat() if invoice.issued_at else None,
        "due_date": invoice.due_date.isoformat() if invoice.due_date else None,
        "sent_at": invoice.sent_at.isoformat() if invoice.sent_at else None,
        "paid_at": invoice.paid_at.isoformat() if invoice.paid_at else None,
        "period_month": invoice.period_month,
        "period_year": invoice.period_year,
        "payments": [_serialize_payment(p) for p in payments],
        "reminders": [_serialize_reminder(r) for r in reminders],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulte une facture (ops)")
    parser.add_argument("invoice_number", nargs="?", help="Ex: EM-2026-02-0034")
    parser.add_argument("--id", type=int, dest="invoice_id", help="ID facture (ex: 373)")
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    args = parser.parse_args()

    if not args.invoice_number and args.invoice_id is None:
        parser.error("Indiquer un numéro de facture ou --id")

    app = create_app()
    with app.app_context():
        row = lookup_invoice(
            invoice_number=args.invoice_number,
            invoice_id=args.invoice_id,
        )

    if row is None:
        label = args.invoice_number or f"id={args.invoice_id}"
        print(f"Facture introuvable: {label}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0

    print(f"Facture {row['invoice_number']} (id={row['id']})")
    print(f"  company_id     : {row['company_id']}")
    print(f"  client         : {row['client_name'] or '-'} (id={row['client_id']})")
    print(f"  bill_to_client : {row['bill_to_client_name'] or '-'} (id={row['bill_to_client_id']})")
    print(f"  status         : {row['status']}")
    print(f"  total / paid / due : {row['total_amount']} / {row['amount_paid']} / {row['balance_due']}")
    print(f"  issued / due   : {row['issued_at']} / {row['due_date']}")
    print(f"  sent / paid    : {row['sent_at']} / {row['paid_at']}")
    print(f"  period         : {row['period_month']}/{row['period_year']}")

    print(f"\n  Paiements ({len(row['payments'])}):")
    if not row["payments"]:
        print("    (aucun)")
    for p in row["payments"]:
        print(
            f"    #{p['id']} {p['amount']} CHF — {p['method']} "
            f"le {p['paid_at']} ref={p['reference'] or '-'}"
        )

    print(f"\n  Rappels ({len(row['reminders'])}):")
    if not row["reminders"]:
        print("    (aucun)")
    for r in row["reminders"]:
        print(
            f"    niveau {r['level']} — total {r['total_due']} CHF "
            f"(principal {r['principal_amount']} + frais {r['reminder_fee_amount']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
