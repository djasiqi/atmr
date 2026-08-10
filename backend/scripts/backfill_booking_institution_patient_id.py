#!/usr/bin/env python3
"""Backfill Booking.institution_patient_id depuis TransportRequest.

Usage (Docker) :
  python scripts/backfill_booking_institution_patient_id.py --dry-run
  python scripts/backfill_booking_institution_patient_id.py --apply

La règle de résolution (demande directe, parent A/R, ``route_group_id``) est
partagée avec la lecture des opportunités de facturation, via
``application.invoices.institution_patient_resolution``.
Les ambiguïtés (plusieurs patients candidats) ne sont jamais écrites.
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill_institution_patient")

BATCH_SIZE = 1000


def _run(*, apply: bool) -> int:
    from app import create_app
    from application.invoices.institution_patient_resolution import (
        build_institution_patient_mapping,
    )
    from ext import db
    from models import Booking

    app = create_app()
    with app.app_context():
        rows = (
            db.session.query(
                Booking.id,
                Booking.parent_booking_id,
                Booking.route_group_id,
                Booking.billing_party_id,
            )
            .filter(Booking.institution_patient_id.is_(None))
            .all()
        )
        logger.info("bookings sans institution_patient_id: %s", len(rows))

        total_resolved = 0
        total_ambiguous = 0
        for start in range(0, len(rows), BATCH_SIZE):
            chunk = rows[start : start + BATCH_SIZE]
            resolved, ambiguous = build_institution_patient_mapping(
                [int(r[0]) for r in chunk],
                parent_ids_by_booking={
                    int(r[0]): int(r[1]) for r in chunk if r[1] is not None
                },
                route_group_by_booking={int(r[0]): str(r[2]) for r in chunk if r[2]},
                billing_party_by_booking={
                    int(r[0]): int(r[3]) for r in chunk if r[3] is not None
                },
            )
            total_resolved += len(resolved)
            total_ambiguous += len(ambiguous)
            for booking_id in sorted(ambiguous):
                logger.warning("needs_review booking_id=%s", booking_id)

            if apply and resolved:
                for booking_id, patient_id in resolved.items():
                    Booking.query.filter_by(id=booking_id).update(
                        {"institution_patient_id": patient_id},
                        synchronize_session=False,
                    )
                db.session.commit()

        logger.info(
            "dry_run=%s resolved=%s ambiguous=%s",
            not apply,
            total_resolved,
            total_ambiguous,
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true", help="Écrit en base")
    args = parser.parse_args()
    return _run(apply=bool(args.apply))


if __name__ == "__main__":
    sys.exit(main())
