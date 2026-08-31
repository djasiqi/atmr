from sqlalchemy import text
from app import create_app
from models import Booking, db
from models.enums import BookingStatus

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
    b = db.session.get(Booking, bid)
    prev = str(b.status)
    b.status = BookingStatus.EN_ROUTE
    db.session.commit()
    try:
        from shared.notifications import notify_booking_update

        notify_booking_update(driver_id=DRIVER_ID, booking=b)
        print("FANOUT ok")
    except Exception as exc:
        print("FANOUT_FAIL", type(exc).__name__, str(exc)[:120])
    print(f"TRANSITION mission_id={bid} {prev} -> {b.status}")
