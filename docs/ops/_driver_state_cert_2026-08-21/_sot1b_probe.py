from datetime import UTC, datetime
from decimal import Decimal
from sqlalchemy import text
from app import create_app
from models import Booking, db
from models.enums import BookingStatus
from application.companies.assignment_binding import ensure_booking_assignment

DRIVER_ID = 20
app = create_app()
with app.app_context():
    d = db.session.execute(text("SELECT id, company_id FROM driver WHERE id=:did"), {"did": DRIVER_ID}).mappings().first()
    company_id = int(d["company_id"])
    client = db.session.execute(
        text("SELECT c.id AS client_id, c.user_id FROM client c WHERE c.company_id=:cid AND c.is_active IS TRUE ORDER BY c.id LIMIT 1"),
        {"cid": company_id},
    ).mappings().first()
    if not client:
        client = db.session.execute(
            text("SELECT c.id AS client_id, c.user_id FROM client c WHERE c.is_active IS TRUE ORDER BY c.id LIMIT 1")
        ).mappings().first()
    print("CLIENT", dict(client) if client else None, "CO", company_id)
    now = datetime.now(UTC)
    b = Booking()
    b.user_id = int(client["user_id"])
    b.company_id = company_id
    b.executing_company_id = company_id
    b.client_id = int(client["client_id"])
    b.driver_id = DRIVER_ID
    b.customer_name = "SOT1B-INVARIANT-PROBE"
    b.pickup_location = "Geneva"
    b.dropoff_location = "HUG"
    b.scheduled_time = now
    b.status = BookingStatus.ASSIGNED
    b.amount = Decimal("1.00")
    b.billed_to_type = "patient"
    b.time_confirmed = True
    db.session.add(b)
    db.session.flush()
    ensure_booking_assignment(company_id=company_id, booking=b, driver_id=DRIVER_ID)
    db.session.commit()
    a = db.session.execute(
        text("SELECT id, status::text AS status, driver_id FROM assignment WHERE booking_id=:id"),
        {"id": b.id},
    ).mappings().first()
    print("PROBE_BOOKING", b.id, "ASSIGNMENT", dict(a) if a else None)
    db.session.execute(text("DELETE FROM assignment WHERE booking_id=:id"), {"id": b.id})
    db.session.execute(text("UPDATE booking SET status='CANCELED' WHERE id=:id"), {"id": b.id})
    db.session.commit()
    print("PROBE_CLEANED OK" if a else "PROBE_FAIL_NO_ASSIGNMENT")
