#!/usr/bin/env python3
"""Rapport écarts Booking vs Assignment (PR6) — dry-run, sortie CSV sur stdout.

Usage::
    python -m scripts.report_booking_assignment_drift [--days 7]

Nécessite ``DATABASE_URL`` / config Flask comme les autres scripts backend.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import or_

# Permet ``python scripts/report_booking_assignment_drift.py`` depuis backend/
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))

from ext import app
from models import Assignment, Booking
from models.enums import AssignmentStatus, BookingStatus

_ACTIVE_BOOKING = (
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE,
    BookingStatus.IN_PROGRESS,
)

# Mission « en cours » côté dispatch (aligné services géoloc / trajet)
_ACTIVE_ASSIGNMENT = (
    AssignmentStatus.EN_ROUTE_PICKUP,
    AssignmentStatus.ARRIVED_PICKUP,
    AssignmentStatus.ONBOARD,
    AssignmentStatus.EN_ROUTE_DROPOFF,
    AssignmentStatus.ARRIVED_DROPOFF,
)


def run_report(days: int) -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=days)
    rows: list[dict[str, Any]] = []

    with app.app_context():
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
            a = Assignment.query.filter(
                Assignment.booking_id == b.id,
                Assignment.driver_id == did,
                Assignment.status.in_(_ACTIVE_ASSIGNMENT),
            ).first()
            mismatch = a is None
            rows.append(
                {
                    "booking_id": b.id,
                    "driver_id": did,
                    "booking_status": getattr(b.status, "value", str(b.status)),
                    "assignment_id": getattr(a, "id", None) if a else None,
                    "assignment_status": (
                        getattr(a.status, "value", str(a.status)) if a else ""
                    ),
                    "mismatch_active_booking_no_active_assignment": mismatch,
                }
            )

    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Rapport dérive Booking vs Assignment")
    p.add_argument("--days", type=int, default=7, help="Fenêtre depuis maintenant (jours)")
    args = p.parse_args()

    rows = run_report(args.days)
    w = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "booking_id",
            "driver_id",
            "booking_status",
            "assignment_id",
            "assignment_status",
            "mismatch_active_booking_no_active_assignment",
        ],
    )
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
