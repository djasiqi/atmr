#!/usr/bin/env python
"""Audit d'intégrité des factures.

Détecte les incohérences dans les factures :
- Totaux à 0 alors que des lignes existent
- subtotal != sum(lines)
- balance_due incohérent
- Factures S2 sans billing_party_id
- Factures cliniques sans lien ClientBillingParty (info seulement)

Usage:
    python scripts/audit_invoices_integrity.py [--company-id ID] [--from-date YYYY-MM-DD] [--verbose]

Exemples:
    # Audit complet
    python scripts/audit_invoices_integrity.py --verbose

    # Audit pour une company spécifique
    python scripts/audit_invoices_integrity.py --company-id 1 --from-date 2026-01-01

    # Export JSON pour réparation
    python scripts/audit_invoices_integrity.py --export audit_results.json
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

# Setup path pour imports
sys.path.insert(0, ".")

from sqlalchemy import text

from app import create_app
from ext import db
from models.enums import BillingPartyType, InvoiceBillingStrategy, InvoiceStatus

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Tolérance pour comparaison de montants (arrondi 5 centimes)
TOLERANCE = Decimal("0.05")

# Valeurs enum pour requêtes SQL
STATUS_DRAFT = InvoiceStatus.DRAFT.value  # "draft"
STATUS_SENT = InvoiceStatus.SENT.value  # "sent"
STATUS_PAID = InvoiceStatus.PAID.value  # "paid"
STATUS_CANCELLED = InvoiceStatus.CANCELLED.value  # "cancelled"
STRATEGY_S2 = InvoiceBillingStrategy.S2_CLINIC_MONTHLY.value  # "s2_clinic_monthly"
BP_TYPE_CLINIC = BillingPartyType.CLINIC.value  # "clinic"
BP_TYPE_EMS = BillingPartyType.EMS.value  # "ems"
BP_TYPE_HOSPITAL = BillingPartyType.HOSPITAL.value  # "hospital"


def parse_args():
    parser = argparse.ArgumentParser(description="Audit d'intégrité des factures")
    parser.add_argument("--company-id", type=int, help="ID de la company à auditer")
    parser.add_argument(
        "--from-date", type=str, help="Date de début (YYYY-MM-DD), filtre sur issued_at"
    )
    parser.add_argument("--to-date", type=str, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument(
        "--status", type=str, help="Filtrer par statut (draft, sent, paid, cancelled)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Afficher les détails"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Nombre max d'exemples par catégorie"
    )
    parser.add_argument("--export", type=str, help="Exporter les résultats en JSON")
    return parser.parse_args()


def build_where_clause(filters: dict) -> tuple[str, dict]:
    """Construit la clause WHERE et les paramètres pour les filtres."""
    clauses = []
    params = {}

    if filters.get("company_id"):
        clauses.append("i.company_id = :company_id")
        params["company_id"] = filters["company_id"]

    if filters.get("from_date"):
        clauses.append("i.issued_at >= :from_date")
        params["from_date"] = filters["from_date"]

    if filters.get("to_date"):
        clauses.append("i.issued_at <= :to_date")
        params["to_date"] = filters["to_date"]

    if filters.get("status"):
        clauses.append("i.status = :status")
        params["status"] = filters["status"]

    where = " AND ".join(clauses) if clauses else "1=1"
    return where, params


def audit_totals_zero_with_lines(session, filters: dict, limit: int) -> list[dict]:
    """Trouve les factures avec total=0 mais des lignes avec montants > 0."""
    where_clause, params = build_where_clause(filters)
    params["limit"] = limit
    params["status_paid"] = STATUS_PAID
    params["status_cancelled"] = STATUS_CANCELLED

    query = text(f"""
        SELECT
            i.id,
            i.invoice_number,
            i.company_id,
            i.status,
            i.billing_strategy,
            i.subtotal_amount,
            i.total_amount,
            i.balance_due,
            COALESCE(SUM(il.line_total), 0) as lines_sum,
            COUNT(il.id) as lines_count
        FROM invoices i
        LEFT JOIN invoice_lines il ON il.invoice_id = i.id
        WHERE {where_clause}
        AND i.total_amount = 0
        AND i.subtotal_amount = 0
        AND i.status NOT IN (:status_paid, :status_cancelled)
        GROUP BY i.id
        HAVING COALESCE(SUM(il.line_total), 0) > 0
        ORDER BY COALESCE(SUM(il.line_total), 0) DESC
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [dict(row._mapping) for row in result]


def audit_subtotal_mismatch(session, filters: dict, limit: int) -> list[dict]:
    """Trouve les factures où subtotal != sum(lines)."""
    where_clause, params = build_where_clause(filters)
    params["limit"] = limit
    params["tolerance"] = float(TOLERANCE)

    query = text(f"""
        SELECT
            i.id,
            i.invoice_number,
            i.company_id,
            i.status,
            i.subtotal_amount,
            COALESCE(SUM(il.line_total), 0) as lines_sum,
            ABS(i.subtotal_amount - COALESCE(SUM(il.line_total), 0)) as diff
        FROM invoices i
        LEFT JOIN invoice_lines il ON il.invoice_id = i.id
        WHERE {where_clause}
        GROUP BY i.id
        HAVING ABS(i.subtotal_amount - COALESCE(SUM(il.line_total), 0)) > :tolerance
        ORDER BY ABS(i.subtotal_amount - COALESCE(SUM(il.line_total), 0)) DESC
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [dict(row._mapping) for row in result]


def audit_balance_due_mismatch(session, filters: dict, limit: int) -> list[dict]:
    """Trouve les factures où balance_due != total_amount - amount_paid."""
    where_clause, params = build_where_clause(filters)
    params["limit"] = limit
    params["tolerance"] = float(TOLERANCE)

    query = text(f"""
        SELECT
            i.id,
            i.invoice_number,
            i.company_id,
            i.status,
            i.total_amount,
            COALESCE(i.amount_paid, 0) as amount_paid,
            i.balance_due,
            (i.total_amount - COALESCE(i.amount_paid, 0)) as expected_balance,
            ABS(i.balance_due - (i.total_amount - COALESCE(i.amount_paid, 0))) as diff
        FROM invoices i
        WHERE {where_clause}
        AND ABS(i.balance_due - (i.total_amount - COALESCE(i.amount_paid, 0))) > :tolerance
        ORDER BY ABS(i.balance_due - (i.total_amount - COALESCE(i.amount_paid, 0))) DESC
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [dict(row._mapping) for row in result]


def audit_s2_without_billing_party(session, filters: dict, limit: int) -> list[dict]:
    """Trouve les factures S2 sans billing_party_id."""
    where_clause, params = build_where_clause(filters)
    params["limit"] = limit
    params["strategy_s2"] = STRATEGY_S2

    query = text(f"""
        SELECT
            i.id,
            i.invoice_number,
            i.company_id,
            i.client_id,
            i.status,
            i.billing_strategy,
            i.billing_party_id
        FROM invoices i
        WHERE {where_clause}
        AND i.billing_strategy = :strategy_s2
        AND i.billing_party_id IS NULL
        ORDER BY i.id DESC
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [dict(row._mapping) for row in result]


def audit_clinic_without_link(session, filters: dict, limit: int) -> list[dict]:
    """Info: factures cliniques sans lien ClientBillingParty (OK avec bypass, mais à surveiller)."""
    where_clause, params = build_where_clause(filters)
    params["limit"] = limit
    params["strategy_s2"] = STRATEGY_S2
    params["bp_type_clinic"] = BP_TYPE_CLINIC
    params["bp_type_ems"] = BP_TYPE_EMS
    params["bp_type_hospital"] = BP_TYPE_HOSPITAL

    query = text(f"""
        SELECT
            i.id,
            i.invoice_number,
            i.company_id,
            i.client_id,
            i.billing_party_id,
            i.billing_strategy,
            bp.type as bp_type,
            bp.display_name as bp_name,
            cbp.id as link_id
        FROM invoices i
        JOIN billing_parties bp ON bp.id = i.billing_party_id
        LEFT JOIN client_billing_parties cbp
            ON cbp.client_id = i.client_id AND cbp.billing_party_id = i.billing_party_id
        WHERE {where_clause}
        AND (
            i.billing_strategy = :strategy_s2
            OR bp.type IN (:bp_type_clinic, :bp_type_ems, :bp_type_hospital)
        )
        AND cbp.id IS NULL
        ORDER BY i.id DESC
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [dict(row._mapping) for row in result]


def audit_negative_balance(session, filters: dict, limit: int) -> list[dict]:
    """Trouve les factures avec balance_due < 0."""
    where_clause, params = build_where_clause(filters)
    params["limit"] = limit

    query = text(f"""
        SELECT
            i.id,
            i.invoice_number,
            i.company_id,
            i.status,
            i.total_amount,
            COALESCE(i.amount_paid, 0) as amount_paid,
            i.balance_due
        FROM invoices i
        WHERE {where_clause}
        AND i.balance_due < 0
        ORDER BY i.balance_due ASC
        LIMIT :limit
    """)
    result = session.execute(query, params)
    return [dict(row._mapping) for row in result]


def print_results(title: str, results: list[dict], verbose: bool = False):
    """Affiche les résultats d'audit."""
    count = len(results)
    status = "⚠️" if count > 0 else "✅"
    print(f"\n{status} {title}: {count} facture(s)")

    if count > 0 and verbose:
        print("-" * 80)
        for i, row in enumerate(results[:10], 1):
            # Formater proprement
            row_str = ", ".join(f"{k}={v}" for k, v in row.items())
            print(f"  {i}. {row_str}")
        if count > 10:
            print(f"  ... et {count - 10} autres")


def serialize_results(results: dict) -> dict:
    """Sérialise les résultats pour export JSON."""

    def convert(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    return {
        k: [{key: convert(val) for key, val in row.items()} for row in v]
        for k, v in results.items()
    }


def main():
    args = parse_args()

    app = create_app()
    with app.app_context():
        filters = {}
        if args.company_id:
            filters["company_id"] = args.company_id
        if args.from_date:
            filters["from_date"] = args.from_date
        if args.to_date:
            filters["to_date"] = args.to_date
        if args.status:
            filters["status"] = args.status

        print("=" * 80)
        print("AUDIT D'INTÉGRITÉ DES FACTURES")
        print("=" * 80)
        print(f"Filtres: {filters or 'Aucun'}")
        print(f"Limite par catégorie: {args.limit}")
        print("Valeurs enum utilisées:")
        print(f"  - STATUS_PAID: {STATUS_PAID}")
        print(f"  - STATUS_CANCELLED: {STATUS_CANCELLED}")
        print(f"  - STRATEGY_S2: {STRATEGY_S2}")

        # Exécuter les audits
        results = {
            "totals_zero_with_lines": audit_totals_zero_with_lines(
                db.session, filters, args.limit
            ),
            "subtotal_mismatch": audit_subtotal_mismatch(
                db.session, filters, args.limit
            ),
            "balance_due_mismatch": audit_balance_due_mismatch(
                db.session, filters, args.limit
            ),
            "s2_without_billing_party": audit_s2_without_billing_party(
                db.session, filters, args.limit
            ),
            "clinic_without_link": audit_clinic_without_link(
                db.session, filters, args.limit
            ),
            "negative_balance": audit_negative_balance(db.session, filters, args.limit),
        }

        # Afficher les résultats
        print_results(
            "Factures avec total=0 mais lignes > 0 (CRITIQUE)",
            results["totals_zero_with_lines"],
            args.verbose,
        )
        print_results(
            "Factures avec subtotal != sum(lines) (CRITIQUE)",
            results["subtotal_mismatch"],
            args.verbose,
        )
        print_results(
            "Factures avec balance_due incohérent (WARNING)",
            results["balance_due_mismatch"],
            args.verbose,
        )
        print_results(
            "Factures S2 sans billing_party_id (CRITIQUE)",
            results["s2_without_billing_party"],
            args.verbose,
        )
        print_results(
            "Factures cliniques sans lien ClientBillingParty (INFO - OK avec bypass)",
            results["clinic_without_link"],
            args.verbose,
        )
        print_results(
            "Factures avec balance_due < 0 (WARNING)",
            results["negative_balance"],
            args.verbose,
        )

        # Résumé
        print("\n" + "=" * 80)
        print("RÉSUMÉ")
        print("=" * 80)

        critical_count = (
            len(results["totals_zero_with_lines"])
            + len(results["subtotal_mismatch"])
            + len(results["s2_without_billing_party"])
        )
        warning_count = len(results["balance_due_mismatch"]) + len(
            results["negative_balance"]
        )

        if critical_count > 0:
            print(
                f"🔴 CRITIQUES: {critical_count} facture(s) nécessitent une correction"
            )
        else:
            print("✅ Aucun problème critique détecté")

        if warning_count > 0:
            print(f"🟡 WARNINGS: {warning_count} facture(s) à vérifier")

        print(
            f"\nℹ️  INFO: {len(results['clinic_without_link'])} facture(s) cliniques sans lien ClientBillingParty"
        )
        print("   (Normal avec le bypass S2/clinic - à ne pas corriger)")

        # Collecter les IDs pour réparation
        repair_ids = {
            "totals_zero": [r["id"] for r in results["totals_zero_with_lines"]],
            "subtotal_mismatch": [r["id"] for r in results["subtotal_mismatch"]],
            "balance_mismatch": [r["id"] for r in results["balance_due_mismatch"]],
            "s2_no_billing_party": [
                r["id"] for r in results["s2_without_billing_party"]
            ],
        }

        print("\n📋 IDs à réparer:")
        print(
            f"   - Totaux à recalculer: {repair_ids['totals_zero'][:10]}{'...' if len(repair_ids['totals_zero']) > 10 else ''}"
        )
        print(
            f"   - Subtotal mismatch: {repair_ids['subtotal_mismatch'][:10]}{'...' if len(repair_ids['subtotal_mismatch']) > 10 else ''}"
        )
        print(
            f"   - Balance mismatch: {repair_ids['balance_mismatch'][:10]}{'...' if len(repair_ids['balance_mismatch']) > 10 else ''}"
        )

        # Export JSON si demandé
        if args.export:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "filters": filters,
                "results": serialize_results(results),
                "repair_ids": repair_ids,
                "summary": {
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                },
            }
            with Path(args.export).open("w") as f:
                json.dump(export_data, f, indent=2)
            print(f"\n📄 Résultats exportés vers: {args.export}")


if __name__ == "__main__":
    main()
