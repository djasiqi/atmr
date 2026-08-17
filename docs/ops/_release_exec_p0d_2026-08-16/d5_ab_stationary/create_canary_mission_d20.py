"""Créer mission canary IN_PROGRESS pour driver 20 (atmr1@atmr.ch) — staging only."""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import text

from app import create_app
from models import Booking, db
from models.enums import BookingStatus

DRIVER_ID = 20


def main() -> None:
    app = create_app()
    with app.app_context():
        d = db.session.execute(
            text("SELECT id, user_id, company_id FROM driver WHERE id=:did"),
            {"did": DRIVER_ID},
        ).mappings().first()
        if not d:
            raise SystemExit("driver 20 missing — run create_canary_driver_b.py first")
        company_id = int(d["company_id"])

        client = db.session.execute(
            text(
                """
                SELECT c.id AS client_id, c.user_id
                FROM client c
                WHERE c.company_id = :cid AND c.is_active IS TRUE
                ORDER BY c.id
                LIMIT 1
                """
            ),
            {"cid": company_id},
        ).mappings().first()
        if not client:
            raise SystemExit("no active client for company")

        closed = db.session.execute(
            text(
                """
                UPDATE booking SET status = 'CANCELED'
                WHERE driver_id = :did
                  AND status::text IN (
                    'ASSIGNED','ACCEPTED','EN_ROUTE','IN_PROGRESS','PENDING'
                  )
                RETURNING id
                """
            ),
            {"did": DRIVER_ID},
        ).fetchall()
        print("CANCELED_PREV", [r[0] for r in closed])

        # Éviter ambiguous_mission si d'autres drivers de la company ont trop d'actives
        # (pas de cancel cross-driver — scope atmr1 seulement)

        now = datetime.now(UTC)
        b = Booking()
        b.user_id = int(client["user_id"])
        b.company_id = company_id
        b.client_id = int(client["client_id"])
        b.driver_id = DRIVER_ID
        b.customer_name = "CANARY-D20-D5-AB-2026-08-16"
        b.pickup_location = "Geneva Gare Cornavin"
        b.dropoff_location = "Hopitaux Universitaires Geneve"
        b.pickup_lat = 46.2102
        b.pickup_lon = 6.1424
        b.dropoff_lat = 46.1936
        b.dropoff_lon = 6.1486
        b.scheduled_time = now
        b.status = BookingStatus.IN_PROGRESS
        b.amount = Decimal("42.00")
        b.billed_to_type = "patient"
        b.time_confirmed = True
        db.session.add(b)
        db.session.commit()
        print(
            "CREATED_MISSION",
            {
                "booking_id": b.id,
                "status": str(b.status),
                "driver_id": b.driver_id,
                "company_id": company_id,
                "scheduled_time": b.scheduled_time.isoformat(),
            },
        )


if __name__ == "__main__":
    main()
