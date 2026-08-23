from application.drivers.update_driver_booking_status import (
    UpdateDriverBookingStatusCommand,
    UpdateDriverBookingStatusUseCase,
)
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from sqlalchemy import text
from app import create_app
from models import db
from shared.notifications import notify_booking_update

DRIVER_ID = 20
app = create_app()
with app.app_context():
    active = db.session.execute(
        text(
            "SELECT id, status::text AS status FROM booking "
            "WHERE driver_id=:did AND status::text IN "
            "('ASSIGNED','ACCEPTED','EN_ROUTE','ARRIVED','IN_PROGRESS') "
            "ORDER BY id DESC LIMIT 5"
        ),
        {"did": DRIVER_ID},
    ).mappings().all()
    print("BEFORE", [dict(r) for r in active])
    if not active:
        raise SystemExit("NO_ACTIVE_MISSION")
    bid = int(active[0]["id"])
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=BookingRepository(),
        assignment_repo=AssignmentRepository(),
        db_session=db.session,
        notify_booking_update_fn=notify_booking_update,
        resolve_delays_fn=lambda *_a, **_k: None,
        emit_assignment_cancelled_fn=lambda *_a, **_k: None,
        maybe_trigger_dispatch_fn=None,
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=bid,
            driver_id=DRIVER_ID,
            payload={"status": "ARRIVED"},
        )
    )
    print("STATUS_CODE", res.status_code)
    print("RESPONSE", res.response)
    # booking reste EN_ROUTE ; milestone ARRIVED
    row = db.session.execute(
        text("SELECT id, status::text AS status FROM booking WHERE id=:id"),
        {"id": bid},
    ).mappings().first()
    print("BOOKING_AFTER", dict(row) if row else None)
