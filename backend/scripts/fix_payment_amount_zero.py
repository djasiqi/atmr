#!/usr/bin/env python3
"""
Script pour corriger les paiements avec amount=0 pour les factures PAID.

Usage:
    python -m scripts.fix_payment_amount_zero [--dry-run] [--limit N]

Options:
    --dry-run: Affiche ce qui serait corrigé sans modifier la DB
    --limit N: Limite le nombre de paiements à corriger (pour tests)
"""

import argparse
import sys
from contextlib import suppress
from decimal import Decimal

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, ".")

from app import create_app
from ext import db
from models.enums import InvoiceStatus
from models.invoice import Invoice, InvoicePayment


def fix_payment_amounts(dry_run=False, limit=None):
    """
    Corrige les paiements avec amount=0 pour les factures marquées PAID.

    Règle:
    - Si invoice.status == PAID et payment.amount == 0
    - Et qu'il n'y a qu'un seul paiement pour cette facture
    - Alors payment.amount = invoice.total_amount
    - Sinon, répartir invoice.amount_paid entre les paiements proportionnellement
    """
    # Trouver les factures PAID avec des paiements amount=0
    query = (
        db.session.query(InvoicePayment)
        .join(Invoice, InvoicePayment.invoice_id == Invoice.id)
        .filter(Invoice.status == InvoiceStatus.PAID)
        .filter(InvoicePayment.amount == 0)
    )

    if limit:
        query = query.limit(limit)

    payments_to_fix = query.all()

    if not payments_to_fix:
        print("✅ Aucun paiement à corriger.")
        return

    print(
        f"🔍 Trouvé {len(payments_to_fix)} paiement(s) avec amount=0 pour des factures PAID"
    )

    fixed_count = 0
    skipped_count = 0

    for payment in payments_to_fix:
        invoice = payment.invoice

        # Vérifier que la facture est bien PAID
        if invoice.status != InvoiceStatus.PAID:
            print(
                f"⚠️  Paiement {payment.id}: facture {invoice.id} n'est pas PAID, ignoré"
            )
            skipped_count += 1
            continue

        # Compter tous les paiements pour cette facture
        all_payments = (
            db.session.query(InvoicePayment)
            .filter(InvoicePayment.invoice_id == invoice.id)
            .all()
        )

        # Si un seul paiement, utiliser le total_amount
        if len(all_payments) == 1:
            new_amount = Decimal(str(invoice.total_amount or 0))
            print(
                f"📝 Paiement {payment.id} (facture {invoice.id}): "
                f"amount=0 → {float(new_amount)} (total_amount)"
            )
            if not dry_run:
                payment.amount = new_amount
            fixed_count += 1
        else:
            # Plusieurs paiements : répartir proportionnellement
            # Calculer la somme des montants non-nuls
            non_zero_payments = [p for p in all_payments if p.amount > 0]
            total_non_zero = sum(Decimal(str(p.amount)) for p in non_zero_payments)
            total_to_distribute = (
                Decimal(str(invoice.amount_paid or 0)) - total_non_zero
            )

            if total_to_distribute > 0:
                # Répartir équitablement entre les paiements à 0
                zero_payments = [p for p in all_payments if p.amount == 0]
                amount_per_payment = total_to_distribute / len(zero_payments)

                for p in zero_payments:
                    print(
                        f"📝 Paiement {p.id} (facture {invoice.id}): "
                        f"amount=0 → {float(amount_per_payment)} (répartition)"
                    )
                    if not dry_run:
                        p.amount = amount_per_payment
                    fixed_count += 1
            else:
                print(
                    f"⚠️  Paiement {payment.id} (facture {invoice.id}): "
                    "impossible de déterminer le montant (plusieurs paiements, amount_paid déjà réparti)"
                )
                skipped_count += 1

    if dry_run:
        print(
            f"\n🔍 DRY-RUN: {fixed_count} paiement(s) seraient corrigés, {skipped_count} ignoré(s)"
        )
    elif fixed_count > 0:
        db.session.commit()
        print(f"\n✅ {fixed_count} paiement(s) corrigé(s), {skipped_count} ignoré(s)")
    else:
        print(f"\n⚠️  Aucun paiement corrigé, {skipped_count} ignoré(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Corrige les paiements avec amount=0 pour les factures PAID"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait corrigé sans modifier la DB",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limite le nombre de paiements à corriger (pour tests)",
    )

    args = parser.parse_args()

    # Créer l'application Flask et le contexte
    app = create_app()

    try:
        with app.app_context():
            fix_payment_amounts(dry_run=args.dry_run, limit=args.limit)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        if app.app_context:
            with suppress(Exception):
                db.session.rollback()
        sys.exit(1)


if __name__ == "__main__":
    main()
