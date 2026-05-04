"""Backfill: remplir Invoice.billing_party_id depuis le legacy bill_to_client_id.

Usage (dans le conteneur API) :
  python scripts/backfill_invoice_billing_party.py --dry-run
  python scripts/backfill_invoice_billing_party.py --limit 500
  python scripts/backfill_invoice_billing_party.py --batch-size 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports (pattern repo)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from ext import db
from models import Invoice
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_legacy_bill_to_client,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run(*, dry_run: bool, limit: int | None, batch_size: int) -> int:
    q = (
        Invoice.query.filter(Invoice.billing_party_id.is_(None))
        .filter(Invoice.bill_to_client_id.isnot(None))
        .filter(Invoice.bill_to_client_id != Invoice.client_id)
        .order_by(Invoice.id.asc())
    )
    if limit:
        q = q.limit(limit)

    invoices = q.all()
    if not invoices:
        logger.info("Aucune facture à backfill.")
        return 0

    logger.info("Factures à traiter: %s (dry_run=%s)", len(invoices), dry_run)

    updated = 0
    for idx, inv in enumerate(invoices, 1):
        bp = get_or_create_billing_party_for_legacy_bill_to_client(
            company_id=inv.company_id, bill_to_client_id=int(inv.bill_to_client_id)
        )
        if bp is None:
            continue
        inv.billing_party_id = bp.id
        updated += 1

        if idx % batch_size == 0:
            if dry_run:
                db.session.rollback()
                logger.info("[dry-run] batch rollback (%s)", idx)
            else:
                db.session.commit()
                logger.info("batch commit (%s) updated=%s", idx, updated)

    if dry_run:
        db.session.rollback()
        logger.info("[dry-run] rollback final. updated=%s", updated)
    else:
        db.session.commit()
        logger.info("commit final. updated=%s", updated)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Ne pas committer")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        run(dry_run=bool(args.dry_run), limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
