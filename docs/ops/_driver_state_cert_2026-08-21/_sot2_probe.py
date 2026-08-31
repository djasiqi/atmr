from datetime import UTC, datetime
from decimal import Decimal
from sqlalchemy import text
from app import create_app
from models import Booking, db
from models.enums import BookingStatus, AssignmentStatus
from application.companies.assignment_binding import ensure_booking_assignment
from application.drivers.compose_driver_mission_surface import compose_driver_mission_payload
from repositories.assignment_repository import AssignmentRepository

DRIVER_ID = 20
app = create_app()
with app.app_context():
    d = db.session.execute(text("SELECT id, company_id FROM driver WHERE id=:did"), {"did": DRIVER_ID}).mappings().first()
    company_id = int(d["company_id"])
    client = db.session.execute(text("SELECT c.id AS client_id, c.user_id FROM client c WHERE c.is_active IS TRUE ORDER BY c.id LIMIT 1")).mappings().first()
    now = datetime.now(UTC)
    b = Booking()
    b.user_id = int(client["user_id"]); b.company_id = company_id; b.executing_company_id = company_id
    b.client_id = int(client["client_id"]); b.driver_id = DRIVER_ID
    b.customer_name = "SOT2-COMPOSE-PROBE"
    b.pickup_location = "Geneva"; b.dropoff_location = "HUG"
    b.scheduled_time = now; b.status = BookingStatus.EN_ROUTE
    b.amount = Decimal("1.00"); b.billed_to_type = "patient"; b.time_confirmed = True
    db.session.add(b); db.session.flush()
    ensure_booking_assignment(company_id=company_id, booking=b, driver_id=DRIVER_ID)
    a = AssignmentRepository().find_model_by_booking_id(int(b.id))
    a.status = AssignmentStatus.ARRIVED_PICKUP
    db.session.commit()
    payload = dict(b.serialize)
    composed = compose_driver_mission_payload(payload, assignment_status=a.status)
    print("BOOKING_DB", b.id, b.status.value)
    print("ASSIGNMENT", a.id, a.status.value)
    print("COMPOSED", composed.get("status"), composed.get("mission_milestone"))
    # Also exercise list helper if importable
    from routes.driver import _compose_driver_bookings_with_assignments
    batch = _compose_driver_bookings_with_assignments([dict(b.serialize)])
    print("BATCH", batch[0].get("status"), batch[0].get("mission_milestone"))
    # cleanup
    db.session.execute(text("DELETE FROM assignment WHERE booking_id=:id"), {"id": b.id})
    db.session.execute(text("UPDATE booking SET status='CANCELED' WHERE id=:id"), {"id": b.id})
    db.session.commit()
    print("SOT2_PROBE_OK" if composed.get("status")=="arrived" and batch[0].get("status")=="arrived" else "SOT2_PROBE_FAIL")
