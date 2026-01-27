"""Script optionnel pour backfill des références QR manquantes.

Ce script remplit invoice.qr_reference pour les factures existantes qui n'ont pas
encore de référence QR générée.

Usage:
    python -m scripts.backfill_qr_references [--dry-run] [--limit N]

Options:
    --dry-run: Affiche ce qui serait fait sans modifier la DB
    --limit N: Traite au maximum N factures (utile pour tester)
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from ext import db
from models import Invoice
from services.documents.qrbill import QRBillService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def backfill_qr_references(dry_run: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Remplit les références QR manquantes pour les factures existantes.

    Args:
        dry_run: Si True, n'effectue pas les modifications en DB
        limit: Nombre maximum de factures à traiter (None = toutes)

    Returns:
        dict avec statistiques (processed, updated, errors, skipped)
    """
    stats = {
        "processed": 0,
        "updated": 0,
        "errors": 0,
        "skipped": 0,
    }

    try:
        # Récupérer les factures sans qr_reference
        query = db.session.query(Invoice).filter(
            Invoice.qr_reference.is_(None)
        )

        if limit:
            query = query.limit(limit)

        invoices = query.all()

        logger.info("Trouvé %s facture(s) sans qr_reference", len(invoices))

        qrbill_service = QRBillService()

        for invoice in invoices:
            stats["processed"] += 1

            try:
                # Générer la référence QR
                qr_ref = qrbill_service._get_payment_reference(invoice)

                if not qr_ref:
                    logger.warning(
                        "Facture %s (%s): Impossible de générer qr_reference (mode NONE ou erreur)",
                        invoice.id,
                        invoice.invoice_number,
                    )
                    stats["skipped"] += 1
                    continue

                if dry_run:
                    logger.info(
                        "[DRY-RUN] Facture %s (%s): Serait mise à jour avec qr_reference=%s",
                        invoice.id,
                        invoice.invoice_number,
                        qr_ref,
                    )
                    stats["updated"] += 1
                else:
                    # Mettre à jour la facture
                    invoice.qr_reference = qr_ref
                    logger.info(
                        "Facture %s (%s): qr_reference=%s",
                        invoice.id,
                        invoice.invoice_number,
                        qr_ref,
                    )
                    stats["updated"] += 1

            except ValueError as e:
                logger.error(
                    "Facture %s (%s): Erreur génération qr_reference: %s",
                    invoice.id,
                    invoice.invoice_number,
                    e,
                )
                stats["errors"] += 1
            except Exception as e:
                logger.exception(
                    "Facture %s (%s): Erreur inattendue: %s",
                    invoice.id,
                    invoice.invoice_number,
                    e,
                )
                stats["errors"] += 1

        if not dry_run and stats["updated"] > 0:
            # Commit toutes les modifications
            db.session.commit()
            logger.info("✅ %s facture(s) mise(s) à jour", stats["updated"])

    except Exception as e:
        logger.exception("Erreur fatale lors du backfill: %s", e)
        if not dry_run:
            db.session.rollback()
        stats["errors"] += 1

    return stats


def main() -> int:
    """Point d'entrée du script."""
    parser = argparse.ArgumentParser(
        description="Backfill des références QR manquantes pour les factures existantes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait fait sans modifier la DB",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximum de factures à traiter (utile pour tester)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Backfill des références QR")
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("⚠️  MODE DRY-RUN (aucune modification en DB)")
    if args.limit:
        logger.info("Limite: %s facture(s)", args.limit)

    stats = backfill_qr_references(dry_run=args.dry_run, limit=args.limit)

    logger.info("=" * 60)
    logger.info("Résumé:")
    logger.info("  Traitées: %s", stats["processed"])
    logger.info("  Mises à jour: %s", stats["updated"])
    logger.info("  Ignorées: %s", stats["skipped"])
    logger.info("  Erreurs: %s", stats["errors"])
    logger.info("=" * 60)

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
