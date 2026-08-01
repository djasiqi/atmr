#!/usr/bin/env python3
"""Backfill Booking.institution_patient_id depuis TransportRequest.

Usage (Docker) :
  python scripts/backfill_booking_institution_patient_id.py --dry-run
  python scripts/backfill_booking_institution_patient_id.py --apply

Ordre :
  1) TR.booking_id → booking
  2) enfants parent_booking_id
  3) même route_group_id
Ambiguïtés (plusieurs patient_id pour un booking) → needs_review (non écrit).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill_institution_patient")


def _run(*, apply: bool) -> int:
    from app import create_app
    from ext import db
    from models import Booking, TransportRequest

    app = create_app()
    with app.app_context():
        # 1) Mapping direct TR.booking_id
        direct_rows = (
            db.session.query(TransportRequest.booking_id, TransportRequest.patient_id)
            .filter(
                TransportRequest.booking_id.isnot(None),
                TransportRequest.patient_id.isnot(None),
            )
            .all()
        )
        by_booking: dict[int, set[int]] = defaultdict(set)
        for bid, pid in direct_rows:
            by_booking[int(bid)].add(int(pid))

        # 2) Propagation parent → enfants
        child_rows = (
            db.session.query(Booking.id, Booking.parent_booking_id)
            .filter(Booking.parent_booking_id.isnot(None))
            .all()
        )
        for cid, pid in child_rows:
            parent_patients = by_booking.get(int(pid), set())
            if parent_patients:
                by_booking[int(cid)].update(parent_patients)

        # 3) route_group_id : TR → bookings du groupe
        tr_groups = (
            db.session.query(TransportRequest.route_group_id, TransportRequest.patient_id)
            .filter(
                TransportRequest.route_group_id.isnot(None),
                TransportRequest.patient_id.isnot(None),
            )
            .all()
        )
        group_patients: dict[str, set[int]] = defaultdict(set)
        for rgid, pid in tr_groups:
            group_patients[str(rgid)].add(int(pid))

        if group_patients:
            group_bookings = (
                db.session.query(Booking.id, Booking.route_group_id)
                .filter(Booking.route_group_id.in_(list(group_patients.keys())))
                .all()
            )
            for bid, rgid in group_bookings:
                pts = group_patients.get(str(rgid), set())
                if pts:
                    by_booking[int(bid)].update(pts)

        resolved = 0
        ambiguous = 0
        skipped_already = 0
        updates: list[tuple[int, int]] = []

        booking_ids = list(by_booking.keys())
        existing = {
            int(b.id): b.institution_patient_id
            for b in Booking.query.filter(Booking.id.in_(booking_ids)).all()
        } if booking_ids else {}

        for bid, pids in sorted(by_booking.items()):
            if bid not in existing:
                continue
            if existing[bid] is not None:
                skipped_already += 1
                continue
            if len(pids) != 1:
                ambiguous += 1
                logger.warning(
                    "needs_review booking_id=%s patient_ids=%s", bid, sorted(pids)
                )
                continue
            pid = next(iter(pids))
            updates.append((bid, pid))
            resolved += 1

        logger.info(
            "dry_run=%s resolved=%s ambiguous=%s already_set=%s",
            not apply,
            resolved,
            ambiguous,
            skipped_already,
        )

        if apply and updates:
            for bid, pid in updates:
                Booking.query.filter_by(id=bid).update(
                    {"institution_patient_id": pid}, synchronize_session=False
                )
            db.session.commit()
            logger.info("applied %s updates", len(updates))
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true", help="Écrit en base")
    args = parser.parse_args()
    apply = bool(args.apply)
    return _run(apply=apply)


if __name__ == "__main__":
    sys.exit(main())
