#!/usr/bin/env python
"""Réparation d'intégrité des factures.

Corrige les incohérences détectées par audit_invoices_integrity.py :
- Recalcule les totaux via recompute_invoice_totals()
- Corrige balance_due si incohérent

⚠️ IMPORTANT: Toujours faire un backup DB avant d'exécuter avec --fix

Usage:
    # Dry-run (défaut) - preview des corrections
    python scripts/repair_invoices_integrity.py --from-date 2026-01-01

    # Exécution réelle
    python scripts/repair_invoices_integrity.py --from-date 2026-01-01 --fix

    # Inclure les factures SENT (risqué, à utiliser avec précaution)
    python scripts/repair_invoices_integrity.py --fix --include-sent

    # Réparer des IDs spécifiques
    python scripts/repair_invoices_integrity.py --invoice-ids 342,345,350 --fix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from sqlalchemy import text

from app import create_app
from ext import db
from infrastructure.invoices.invoice_calculator import recompute_invoice_totals
from models import Invoice
from models.enums import InvoiceStatus

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Valeurs enum pour requêtes SQL
STATUS_DRAFT = InvoiceStatus.DRAFT.value  # "draft"
STATUS_SENT = InvoiceStatus.SENT.value  # "sent"
STATUS_PAID = InvoiceStatus.PAID.value  # "paid"
STATUS_CANCELLED = InvoiceStatus.CANCELLED.value  # "cancelled"


def parse_args():
    parser = argparse.ArgumentParser(description="Réparation d'intégrité des factures")
    parser.add_argument("--company-id", type=int, help="ID de la company")
    parser.add_argument("--from-date", type=str, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument(
        "--invoice-ids", type=str, help="IDs spécifiques (ex: 342,345,350)"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Exécuter les corrections (sinon dry-run)"
    )
    parser.add_argument(
        "--include-sent", action="store_true", help="Inclure les factures SENT (risqué)"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Nombre max de factures à corriger"
    )
    parser.add_argument("--export", type=str, help="Exporter le rapport en JSON")
    return parser.parse_args()


def build_status_filter(include_sent: bool) -> tuple[str, dict]:
    """Construit le filtre de statut."""
    if include_sent:
        # Uniquement exclure PAID et CANCELLED
        return "i.status NOT IN (:status_paid, :status_cancelled)", {
            "status_paid": STATUS_PAID,
            "status_cancelled": STATUS_CANCELLED,
        }
    # Par défaut, uniquement DRAFT
    return "i.status = :status_draft", {
        "status_draft": STATUS_DRAFT,
    }


def get_invoices_with_zero_totals(
    session, filters: dict, include_sent: bool, limit: int
) -> list[int]:
    """Récupère les IDs des factures avec total=0 mais des lignes > 0."""
    status_clause, status_params = build_status_filter(include_sent)

    where_parts = [status_clause]
    params = {**status_params, "limit": limit}

    if filters.get("company_id"):
        where_parts.append("i.company_id = :company_id")
        params["company_id"] = filters["company_id"]

    if filters.get("from_date"):
        where_parts.append("i.issued_at >= :from_date")
        params["from_date"] = filters["from_date"]

    if filters.get("to_date"):
        where_parts.append("i.issued_at <= :to_date")
        params["to_date"] = filters["to_date"]

    if filters.get("invoice_ids"):
        where_parts.append("i.id = ANY(:invoice_ids)")
        params["invoice_ids"] = filters["invoice_ids"]

    where_clause = " AND ".join(where_parts)

    query = text(f"""
        SELECT i.id
        FROM invoices i
        LEFT JOIN invoice_lines il ON il.invoice_id = i.id
        WHERE {where_clause}
        AND i.total_amount = 0
        AND i.subtotal_amount = 0
        GROUP BY i.id
        HAVING COALESCE(SUM(il.line_total), 0) > 0
        ORDER BY i.id
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [row[0] for row in result]


def get_invoices_with_subtotal_mismatch(
    session, filters: dict, include_sent: bool, limit: int
) -> list[int]:
    """Récupère les IDs des factures où subtotal != sum(lines)."""
    status_clause, status_params = build_status_filter(include_sent)

    where_parts = [status_clause]
    params = {**status_params, "limit": limit, "tolerance": 0.05}

    if filters.get("company_id"):
        where_parts.append("i.company_id = :company_id")
        params["company_id"] = filters["company_id"]

    if filters.get("from_date"):
        where_parts.append("i.issued_at >= :from_date")
        params["from_date"] = filters["from_date"]

    if filters.get("to_date"):
        where_parts.append("i.issued_at <= :to_date")
        params["to_date"] = filters["to_date"]

    if filters.get("invoice_ids"):
        where_parts.append("i.id = ANY(:invoice_ids)")
        params["invoice_ids"] = filters["invoice_ids"]

    where_clause = " AND ".join(where_parts)

    query = text(f"""
        SELECT i.id
        FROM invoices i
        LEFT JOIN invoice_lines il ON il.invoice_id = i.id
        WHERE {where_clause}
        GROUP BY i.id
        HAVING ABS(i.subtotal_amount - COALESCE(SUM(il.line_total), 0)) > :tolerance
        ORDER BY i.id
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [row[0] for row in result]


def get_invoices_with_balance_mismatch(
    session, filters: dict, include_sent: bool, limit: int
) -> list[int]:
    """Récupère les IDs des factures avec balance_due incohérent."""
    status_clause, status_params = build_status_filter(include_sent)

    where_parts = [status_clause]
    params = {**status_params, "limit": limit, "tolerance": 0.01}

    if filters.get("company_id"):
        where_parts.append("i.company_id = :company_id")
        params["company_id"] = filters["company_id"]

    if filters.get("from_date"):
        where_parts.append("i.issued_at >= :from_date")
        params["from_date"] = filters["from_date"]

    if filters.get("to_date"):
        where_parts.append("i.issued_at <= :to_date")
        params["to_date"] = filters["to_date"]

    if filters.get("invoice_ids"):
        where_parts.append("i.id = ANY(:invoice_ids)")
        params["invoice_ids"] = filters["invoice_ids"]

    where_clause = " AND ".join(where_parts)

    query = text(f"""
        SELECT i.id
        FROM invoices i
        WHERE {where_clause}
        AND ABS(i.balance_due - (i.total_amount - COALESCE(i.amount_paid, 0))) > :tolerance
        ORDER BY i.id
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [row[0] for row in result]


def repair_invoice_totals(invoice_id: int, dry_run: bool = True) -> dict[str, Any]:
    """Recalcule les totaux d'une facture.

    Utilise recompute_invoice_totals() pour garantir la même logique qu'en prod.
    """
    invoice = db.session.get(Invoice, invoice_id)
    if not invoice:
        return {
            "status": "error",
            "invoice_id": invoice_id,
            "reason": "invoice_not_found",
        }

    old_values = {
        "subtotal_amount": float(invoice.subtotal_amount or Decimal("0")),
        "vat_total_amount": float(invoice.vat_total_amount or Decimal("0")),
        "total_amount": float(invoice.total_amount or Decimal("0")),
        "balance_due": float(invoice.balance_due or Decimal("0")),
    }

    # Utiliser la MÊME fonction que la prod (commit=False pour preview)
    result = recompute_invoice_totals(invoice_id, commit=not dry_run)

    if result is None:
        return {
            "status": "error",
            "invoice_id": invoice_id,
            "reason": "recompute_failed",
        }

    new_values = {
        "subtotal_amount": float(result["subtotal"]),
        "vat_total_amount": float(result["vat_total"]),
        "total_amount": float(result["total"]),
        "balance_due": float(result["balance_due"]),  # Utilise le balance_due calculé
        "lines_count": result["lines_count"],
    }

    status = "repaired" if not dry_run else "preview"

    if not dry_run:
        logger.info(
            "[REPAIR] Invoice %s (%s): total %.2f → %.2f (%d lignes)",
            invoice_id,
            invoice.invoice_number,
            old_values["total_amount"],
            new_values["total_amount"],
            new_values["lines_count"],
        )

    return {
        "status": status,
        "invoice_id": invoice_id,
        "invoice_number": invoice.invoice_number,
        "old": old_values,
        "new": new_values,
    }


def repair_balance_due(invoice_id: int, dry_run: bool = True) -> dict[str, Any]:
    """Corrige le balance_due d'une facture."""
    invoice = db.session.get(Invoice, invoice_id)
    if not invoice:
        return {
            "status": "error",
            "invoice_id": invoice_id,
            "reason": "invoice_not_found",
        }

    old_balance = float(invoice.balance_due or Decimal("0"))
    amount_paid = invoice.amount_paid or Decimal("0")
    total_amount = invoice.total_amount or Decimal("0")
    expected_balance = float(total_amount - amount_paid)

    if dry_run:
        return {
            "status": "preview",
            "invoice_id": invoice_id,
            "invoice_number": invoice.invoice_number,
            "old_balance_due": old_balance,
            "new_balance_due": expected_balance,
            "total_amount": float(total_amount),
            "amount_paid": float(amount_paid),
        }
    invoice.balance_due = Decimal(str(expected_balance))
    db.session.commit()
    logger.info(
        "[REPAIR] Invoice %s (%s): balance_due %.2f → %.2f",
        invoice_id,
        invoice.invoice_number,
        old_balance,
        expected_balance,
    )
    return {
        "status": "repaired",
        "invoice_id": invoice_id,
        "invoice_number": invoice.invoice_number,
        "old_balance_due": old_balance,
        "new_balance_due": expected_balance,
    }


def main():
    args = parse_args()
    dry_run = not args.fix

    app = create_app()
    with app.app_context():
        print("=" * 80)
        print("RÉPARATION D'INTÉGRITÉ DES FACTURES")
        print("=" * 80)
        print(f"Mode: {'🔴 EXÉCUTION' if args.fix else '🟡 DRY-RUN (preview)'}")
        print(
            f"Inclure SENT: {'Oui ⚠️' if args.include_sent else 'Non (DRAFT uniquement)'}"
        )
        print(f"Limite: {args.limit} factures max")

        # Construire les filtres
        filters = {}
        if args.company_id:
            filters["company_id"] = args.company_id
        if args.from_date:
            filters["from_date"] = args.from_date
        if args.to_date:
            filters["to_date"] = args.to_date
        if args.invoice_ids:
            filters["invoice_ids"] = [
                int(x.strip()) for x in args.invoice_ids.split(",")
            ]

        print(f"Filtres: {filters or 'Aucun'}")

        if args.fix:
            print("\n⚠️  ATTENTION: Les modifications seront appliquées!")
            print("   Assurez-vous d'avoir fait un backup de la DB.")
            confirm = input("Confirmer (oui/non): ")
            if confirm.lower() != "oui":
                print("Annulé.")
                return

        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "filters": filters,
            "repairs": {
                "totals": [],
                "balance": [],
            },
        }

        # 1. Réparer les totaux à 0
        print("\n" + "-" * 40)
        print("1. Factures avec totaux à 0")
        print("-" * 40)

        invoice_ids = get_invoices_with_zero_totals(
            db.session, filters, args.include_sent, args.limit
        )
        print(f"Factures trouvées: {len(invoice_ids)}")

        for invoice_id in invoice_ids:
            result = repair_invoice_totals(invoice_id, dry_run=dry_run)
            report["repairs"]["totals"].append(result)
            if result["status"] == "preview":
                print(
                    f"  [PREVIEW] #{result.get('invoice_number', invoice_id)}: "
                    f"{result.get('old', {}).get('total_amount', 0):.2f} → "
                    f"{result.get('new', {}).get('total_amount', 0):.2f}"
                )
            elif result["status"] == "repaired":
                print(f"  [✅ REPAIRED] #{result.get('invoice_number', invoice_id)}")
            else:
                print(f"  [❌ ERROR] #{invoice_id}: {result.get('reason', 'unknown')}")

        # 2. Réparer les subtotal mismatch (même logique que totaux à 0)
        print("\n" + "-" * 40)
        print("2. Factures avec subtotal != sum(lines)")
        print("-" * 40)

        invoice_ids = get_invoices_with_subtotal_mismatch(
            db.session, filters, args.include_sent, args.limit
        )
        print(f"Factures trouvées: {len(invoice_ids)}")

        for invoice_id in invoice_ids:
            result = repair_invoice_totals(invoice_id, dry_run=dry_run)
            report["repairs"]["totals"].append(result)
            if result["status"] == "preview":
                print(
                    f"  [PREVIEW] #{result.get('invoice_number', invoice_id)}: "
                    f"subtotal {result.get('old', {}).get('subtotal_amount', 0):.2f} → "
                    f"{result.get('new', {}).get('subtotal_amount', 0):.2f}"
                )
            elif result["status"] == "repaired":
                print(f"  [✅ REPAIRED] #{result.get('invoice_number', invoice_id)}")
            else:
                print(f"  [❌ ERROR] #{invoice_id}: {result.get('reason', 'unknown')}")

        # 3. Réparer les balance_due incohérents
        print("\n" + "-" * 40)
        print("3. Factures avec balance_due incohérent")
        print("-" * 40)

        invoice_ids = get_invoices_with_balance_mismatch(
            db.session, filters, args.include_sent, args.limit
        )
        print(f"Factures trouvées: {len(invoice_ids)}")

        for invoice_id in invoice_ids:
            result = repair_balance_due(invoice_id, dry_run=dry_run)
            report["repairs"]["balance"].append(result)
            if result["status"] == "preview":
                print(
                    f"  [PREVIEW] #{result.get('invoice_number', invoice_id)}: "
                    f"balance_due {result.get('old_balance_due', 0):.2f} → "
                    f"{result.get('new_balance_due', 0):.2f}"
                )
            elif result["status"] == "repaired":
                print(f"  [✅ REPAIRED] #{result.get('invoice_number', invoice_id)}")
            else:
                print(f"  [❌ ERROR] #{invoice_id}: {result.get('reason', 'unknown')}")

        # Résumé
        print("\n" + "=" * 80)
        total_repaired = sum(
            1
            for r in report["repairs"]["totals"] + report["repairs"]["balance"]
            if r.get("status") == "repaired"
        )
        total_preview = sum(
            1
            for r in report["repairs"]["totals"] + report["repairs"]["balance"]
            if r.get("status") == "preview"
        )
        total_errors = sum(
            1
            for r in report["repairs"]["totals"] + report["repairs"]["balance"]
            if r.get("status") == "error"
        )

        if dry_run:
            print(f"✅ DRY-RUN terminé: {total_preview} facture(s) à corriger")
            print("   Exécutez avec --fix pour appliquer les corrections.")
        else:
            print(f"✅ RÉPARATION terminée: {total_repaired} facture(s) corrigée(s)")
            if total_errors > 0:
                print(f"   ⚠️ {total_errors} erreur(s) rencontrée(s)")

        print("=" * 80)

        # Export JSON si demandé
        if args.export:
            with Path(args.export).open("w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Rapport exporté vers: {args.export}")


if __name__ == "__main__":
    main()
