#!/usr/bin/env python3
"""Rétro-alignement AssignmentStatus depuis BookingStatus (missions actives récentes).

Usage::
    docker compose exec backend python -m scripts.retro_sync_assignment_status --days 7 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import or_

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))

from app import create_app
from ext import db
from models import Assignment, Booking
from models.enums import AssignmentStatus, BookingStatus
from services.dispatch.assignment_status_sync import (
    resolve_assignment_status_for_transition,
)

_ACTIVE_BOOKING = (
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE,
    BookingStatus.IN_PROGRESS,
)

_BOOKING_TO_TRANSITION: dict[BookingStatus, str] = {
    BookingStatus.EN_ROUTE: "en_route",
    BookingStatus.IN_PROGRESS: "in_progress",
}


def _target_for_booking(booking: Booking) -> AssignmentStatus | None:
    status = booking.status
    if status == BookingStatus.ASSIGNED:
        return AssignmentStatus.SCHEDULED
    transition = _BOOKING_TO_TRANSITION.get(status)
    if transition is None:
        return None
    return resolve_assignment_status_for_transition(transition)


def run_retro_sync(*, days: int, dry_run: bool) -> int:
    since = datetime.now(UTC) - timedelta(days=days)
    updated = 0
    skipped = 0

    with create_app().app_context():
        bookings = (
            Booking.query.filter(
                Booking.driver_id.isnot(None),
                Booking.status.in_(_ACTIVE_BOOKING),
                or_(Booking.updated_at.is_(None), Booking.updated_at >= since),
            )
            .order_by(Booking.id.desc())
            .limit(5000)
            .all()
        )

        for booking in bookings:
            target = _target_for_booking(booking)
            if target is None:
                skipped += 1
                continue
            assignment = Assignment.query.filter_by(booking_id=booking.id).first()
            if assignment is None:
                skipped += 1
                continue
            current = assignment.status
            if current == target:
                skipped += 1
                continue
            print(
                f"booking_id={booking.id} assignment_id={assignment.id} "
                f"{getattr(current, 'value', current)} → {target.value}"
            )
            if not dry_run:
                assignment.status = target
                assignment.updated_at = datetime.now(UTC)
            updated += 1

        if not dry_run and updated:
            db.session.commit()

    print(f"retro_sync done updated={updated} skipped={skipped} dry_run={dry_run}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Rétro-sync AssignmentStatus")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les changements sans commit",
    )
    args = parser.parse_args()
    run_retro_sync(days=args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
