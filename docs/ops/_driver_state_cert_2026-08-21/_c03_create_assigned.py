from __future__ import annotations
from datetime import UTC, datetime
from decimal import Decimal
from sqlalchemy import text
from app import create_app
from models import Booking, db
from models.enums import BookingStatus

DRIVER_ID = 20
app = create_app()
with app.app_context():
    d = db.session.execute(text("SELECT id, company_id FROM driver WHERE id=:did"), {"did": DRIVER_ID}).mappings().first()
    company_id = int(d["company_id"])
    client = db.session.execute(
        text("SELECT c.id AS client_id, c.user_id, c.company_id FROM client c WHERE c.company_id=:cid AND c.is_active IS TRUE ORDER BY c.id LIMIT 1"),
        {"cid": company_id},
    ).mappings().first()
    if not client:
        client = db.session.execute(
            text("SELECT c.id AS client_id, c.user_id, c.company_id FROM client c WHERE c.is_active IS TRUE ORDER BY c.id LIMIT 1")
        ).mappings().first()
    booking_company_id = int(client["company_id"])
    closed = db.session.execute(
        text("UPDATE booking SET status='CANCELED' WHERE driver_id=:did AND status::text IN ('ASSIGNED','ACCEPTED','EN_ROUTE','ARRIVED','IN_PROGRESS','PENDING') RETURNING id"),
        {"did": DRIVER_ID},
    ).fetchall()
    print("CANCELED_PREV", [r[0] for r in closed])
    now = datetime.now(UTC)
    b = Booking()
    b.user_id = int(client["user_id"])
    b.company_id = booking_company_id
    b.client_id = int(client["client_id"])
    b.driver_id = DRIVER_ID
    b.customer_name = "CANARY-C03-ASSIGNED"
    b.pickup_location = "Geneva Gare Cornavin"
    b.dropoff_location = "Hopitaux Universitaires Geneve"
    b.pickup_lat = 46.2102
    b.pickup_lon = 6.1424
    b.dropoff_lat = 46.1936
    b.dropoff_lon = 6.1486
    b.scheduled_time = now  # dans fenêtre T-30
    b.status = BookingStatus.ASSIGNED
    b.amount = Decimal("42.00")
    b.billed_to_type = "patient"
    b.time_confirmed = True
    db.session.add(b)
    db.session.commit()
    try:
        from shared.notifications import notify_booking_update
        notify_booking_update(driver_id=DRIVER_ID, booking=b)
        print("FANOUT booking_updated ok")
    except Exception as exc:
        print("FANOUT_FAIL", type(exc).__name__, str(exc)[:120])
    print(f"CREATED mission_id={b.id} status={b.status} scheduled={b.scheduled_time.isoformat()}")
