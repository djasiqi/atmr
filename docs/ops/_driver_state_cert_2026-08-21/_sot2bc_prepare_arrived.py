"""Prépare mission ARRIVED durable pour SOT2-B/C (ne pas passer IN_PROGRESS)."""
from datetime import UTC, datetime
from decimal import Decimal
import json
from sqlalchemy import text
from app import create_app
from models import Booking, db
from models.enums import BookingStatus, AssignmentStatus
from application.companies.assignment_binding import ensure_booking_assignment
from repositories.assignment_repository import AssignmentRepository

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
    closed = db.session.execute(
        text("""UPDATE booking SET status='CANCELED'
                WHERE driver_id=:did AND status::text IN
                ('ASSIGNED','ACCEPTED','EN_ROUTE','IN_PROGRESS','PENDING')
                RETURNING id"""),
        {"did": DRIVER_ID},
    ).fetchall()
    print("CANCELED_PREV", [r[0] for r in closed])
    now = datetime.now(UTC)
    b = Booking()
    b.user_id = int(client["user_id"])
    b.company_id = company_id
    b.executing_company_id = company_id
    b.client_id = int(client["client_id"])
    b.driver_id = DRIVER_ID
    b.customer_name = "CANARY-SOT2BC-ARRIVED"
    b.pickup_location = "Geneva Gare Cornavin"
    b.dropoff_location = "Hopitaux Universitaires Geneve"
    b.pickup_lat = 46.2102
    b.pickup_lon = 6.1424
    b.dropoff_lat = 46.1936
    b.dropoff_lon = 6.1486
    b.scheduled_time = now
    b.status = BookingStatus.EN_ROUTE
    b.amount = Decimal("42.00")
    b.billed_to_type = "patient"
    b.time_confirmed = True
    db.session.add(b)
    db.session.flush()
    ensure_booking_assignment(company_id=company_id, booking=b, driver_id=DRIVER_ID)
    a = AssignmentRepository().find_model_by_booking_id(int(b.id))
    a.status = AssignmentStatus.ARRIVED_PICKUP
    db.session.commit()
    out = {
        "mission_id": b.id,
        "booking_status": b.status.value,
        "assignment_id": a.id,
        "assignment_status": a.status.value,
    }
    print("READY", json.dumps(out))
