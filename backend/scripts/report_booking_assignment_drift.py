#!/usr/bin/env python3
"""Rapport écarts Booking vs Assignment (P0-A gate drift).

Usage::
    docker compose exec api python -m scripts.report_booking_assignment_drift --days 7

``Booking ASSIGNED`` + ``Assignment SCHEDULED`` est **cohérent** et n'est pas compté
comme dérive. Seules les incohérences métier actives (ex. ``EN_ROUTE`` sans
``EN_ROUTE_PICKUP``) sont signalées.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import or_

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))

from app import create_app
from models import Assignment, Booking
from models.enums import AssignmentStatus, BookingStatus

_ACTIVE_BOOKING = (
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE,
    BookingStatus.IN_PROGRESS,
)

_TERMINAL_ASSIGNMENT = (
    AssignmentStatus.CANCELLED,
    AssignmentStatus.REASSIGNED,
    AssignmentStatus.COMPLETED,
    AssignmentStatus.NO_SHOW,
)

# Paires booking → assignment considérées saines (aligné assignment_status_sync v1.1)
_EXPECTED_ASSIGNMENT_BY_BOOKING: dict[BookingStatus, frozenset[AssignmentStatus]] = {
    BookingStatus.ASSIGNED: frozenset({AssignmentStatus.SCHEDULED}),
    BookingStatus.EN_ROUTE: frozenset(
        {
            AssignmentStatus.EN_ROUTE_PICKUP,
            AssignmentStatus.ARRIVED_PICKUP,
        }
    ),
    BookingStatus.IN_PROGRESS: frozenset(
        {
            AssignmentStatus.ONBOARD,
            AssignmentStatus.EN_ROUTE_DROPOFF,
            AssignmentStatus.ARRIVED_DROPOFF,
        }
    ),
}


def expected_assignment_statuses(
    booking_status: BookingStatus,
) -> frozenset[AssignmentStatus]:
    """Statuts assignment acceptables pour un booking actif donné."""
    return _EXPECTED_ASSIGNMENT_BY_BOOKING.get(booking_status, frozenset())


def is_status_drift(
    *,
    booking_status: BookingStatus,
    assignment_status: AssignmentStatus | None,
) -> bool:
    """True si la paire booking/assignment est incohérente pour une mission active."""
    if assignment_status is None:
        return True
    expected = expected_assignment_statuses(booking_status)
    if not expected:
        return True
    return assignment_status not in expected


def evaluate_drift_row(
    *,
    booking_id: int,
    driver_id: int,
    booking_status: BookingStatus,
    assignment_id: int | None,
    assignment_status: AssignmentStatus | None,
) -> dict[str, Any]:
    expected = expected_assignment_statuses(booking_status)
    drift = is_status_drift(
        booking_status=booking_status,
        assignment_status=assignment_status,
    )
    return {
        "booking_id": booking_id,
        "driver_id": driver_id,
        "booking_status": getattr(booking_status, "value", str(booking_status)),
        "assignment_id": assignment_id,
        "assignment_status": (
            getattr(assignment_status, "value", str(assignment_status))
            if assignment_status is not None
            else ""
        ),
        "expected_assignment_statuses": "|".join(sorted(s.value for s in expected)),
        "status_drift": drift,
    }


def run_report(days: int) -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=days)
    rows: list[dict[str, Any]] = []

    flask_app = create_app()
    with flask_app.app_context():
        active = (
            Booking.query.filter(
                Booking.driver_id.isnot(None),
                Booking.status.in_(_ACTIVE_BOOKING),
                or_(Booking.updated_at.is_(None), Booking.updated_at >= since),
            )
            .order_by(Booking.id.desc())
            .limit(5000)
            .all()
        )

        for b in active:
            did = b.driver_id
            if not did:
                continue
            booking_status = b.status
            if not isinstance(booking_status, BookingStatus):
                try:
                    booking_status = BookingStatus(booking_status)
                except (ValueError, TypeError):
                    continue

            a = (
                Assignment.query.filter(
                    Assignment.booking_id == b.id,
                    Assignment.driver_id == did,
                    Assignment.status.notin_(_TERMINAL_ASSIGNMENT),
                )
                .order_by(Assignment.id.desc())
                .first()
            )
            assignment_status = a.status if a else None
            if assignment_status is not None and not isinstance(
                assignment_status, AssignmentStatus
            ):
                try:
                    assignment_status = AssignmentStatus(assignment_status)
                except (ValueError, TypeError):
                    assignment_status = None

            rows.append(
                evaluate_drift_row(
                    booking_id=b.id,
                    driver_id=did,
                    booking_status=booking_status,
                    assignment_id=getattr(a, "id", None) if a else None,
                    assignment_status=assignment_status,
                )
            )

    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Rapport dérive Booking vs Assignment")
    p.add_argument(
        "--days", type=int, default=7, help="Fenêtre depuis maintenant (jours)"
    )
    args = p.parse_args()

    rows = run_report(args.days)
    drift_count = sum(1 for r in rows if r["status_drift"])
    print(
        f"[report_booking_assignment_drift] rows={len(rows)} status_drift={drift_count}",
        file=sys.stderr,
    )

    w = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "booking_id",
            "driver_id",
            "booking_status",
            "assignment_id",
            "assignment_status",
            "expected_assignment_statuses",
            "status_drift",
        ],
    )
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return 1 if drift_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
