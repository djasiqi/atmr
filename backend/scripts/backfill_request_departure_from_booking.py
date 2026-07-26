"""Aligne transport_requests.scheduled_time sur le booking principal lié.

Corrige les écarts créés avant la sync automatique (ex. planification
transporteur via ScheduleCompanyReservationUseCase).

Usage (Docker) :
    docker compose run --rm -T atmr_api python -m scripts.backfill_request_departure_from_booking
    docker compose run --rm -T atmr_api python -m scripts.backfill_request_departure_from_booking --dry-run
    docker compose run --rm -T atmr_api python -m scripts.backfill_request_departure_from_booking --request-id 2314
"""

from __future__ import annotations

import argparse
import logging
import sys

from ext import db
from models import Booking, TransportRequest
from services.institutions.mission_schedule import (
    sync_transport_request_departure_from_booking,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def backfill(
    *,
    dry_run: bool = False,
    request_id: int | None = None,
) -> dict[str, int]:
    stats = {"scanned": 0, "updated": 0, "skipped": 0, "errors": 0}

    query = (
        db.session.query(TransportRequest, Booking)
        .join(Booking, TransportRequest.booking_id == Booking.id)
        .filter(Booking.scheduled_time.isnot(None))
    )
    if request_id is not None:
        query = query.filter(TransportRequest.id == request_id)

    rows = query.all()
    logger.info("%d demande(s) convertie(s) avec booking horaire renseigné", len(rows))

    for transport_request, booking in rows:
        stats["scanned"] += 1
        try:
            tr_st = transport_request.scheduled_time
            b_st = booking.scheduled_time
            if tr_st == b_st and transport_request.pickup_time_confirmed:
                stats["skipped"] += 1
                continue

            logger.info(
                "Request #%s / booking #%s : request=%s booking=%s",
                transport_request.id,
                booking.id,
                tr_st,
                b_st,
            )

            if dry_run:
                stats["updated"] += 1
                continue

            if sync_transport_request_departure_from_booking(
                transport_request, booking
            ):
                stats["updated"] += 1
            else:
                stats["skipped"] += 1
        except Exception as exc:
            stats["errors"] += 1
            logger.exception(
                "Échec request #%s booking #%s : %s",
                transport_request.id,
                booking.id,
                exc,
            )

    if not dry_run and stats["updated"] > 0:
        db.session.commit()
        logger.info("Commit effectué (%d mise(s) à jour)", stats["updated"])
    elif dry_run:
        logger.info("Dry-run : aucune écriture en base")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill transport_requests.scheduled_time depuis bookings"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les écarts sans modifier la base",
    )
    parser.add_argument(
        "--request-id",
        type=int,
        default=None,
        help="Limiter à une demande (ex. 2314)",
    )
    args = parser.parse_args()

    from app import create_app

    app = create_app()
    with app.app_context():
        stats = backfill(dry_run=args.dry_run, request_id=args.request_id)
        logger.info("Résultat : %s", stats)
        if stats["errors"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
