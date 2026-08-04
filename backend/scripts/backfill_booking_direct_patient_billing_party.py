#!/usr/bin/env python3
"""Backfill Booking.billing_party_id pour les courses patient portefeuille.

Usage (Docker / conteneur API) :
  python scripts/backfill_booking_direct_patient_billing_party.py --dry-run
  python scripts/backfill_booking_direct_patient_billing_party.py --apply
  python scripts/backfill_booking_direct_patient_billing_party.py --apply --company-id 12
  python scripts/backfill_booking_direct_patient_billing_party.py --apply --limit 500

Règle :
  - billed_to_type=patient, billing_party_id NULL, invoice_line_id NULL
  - tiers ClientBillingParty actif si présent, sinon BillingParty PATIENT
    (external_ref=patient_client:{client_id})
  - aucun ClientBillingParty créé pour le destinataire patient
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from services.billing.direct_patient_billing_party_backfill import (
    backfill_summary_dict,
    run_backfill_direct_patient_billing_party,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rattache un BillingParty aux courses patient sans destinataire V2."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simuler sans commit (défaut si --apply absent)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Appliquer et committer les mises à jour",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de courses")
    parser.add_argument(
        "--company-id",
        type=int,
        default=None,
        help="Restreindre à une entreprise",
    )
    args = parser.parse_args()

    # Défaut : dry-run (sécurité ops)
    dry_run = not bool(args.apply)

    app = create_app()
    with app.app_context():
        result = run_backfill_direct_patient_billing_party(
            dry_run=dry_run,
            limit=args.limit,
            company_id=args.company_id,
        )
        logger.info("Résultat: %s", backfill_summary_dict(result))


if __name__ == "__main__":
    main()
