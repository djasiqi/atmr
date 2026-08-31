from datetime import datetime, timezone
from sqlalchemy import text
from app import create_app
from models import Booking, db
from ext import socketio

DRIVER_ID = 20
BID = 51
app = create_app()
with app.app_context():
    b = db.session.get(Booking, BID)
    payload = b.to_dict() if hasattr(b, "to_dict") else {"id": BID}
    # Forcer milestone + status UX arrived (booking DB reste EN_ROUTE)
    payload["status"] = "arrived"
    payload["mission_milestone"] = "ARRIVED"
    payload["id"] = BID
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["updated_at"] = now
    room = f"driver_{DRIVER_ID}"
    # Format brut booking_updated (comme notify_booking_update)
    socketio.emit("booking_updated", payload, to=room)
    # Format événement canonique driver si le bridge l'attend
    socketio.emit(
        "mission_status_changed",
        {
            "mission_id": BID,
            "event_type": "mission_status_changed",
            "updated_at": now,
            "event_sequence": int(datetime.now(timezone.utc).timestamp()),
            "payload": {"status": "ARRIVED", "mission_milestone": "ARRIVED", "id": BID},
        },
        to=room,
    )
    socketio.emit(
        "mission_updated",
        {
            "mission_id": BID,
            "event_type": "mission_updated",
            "updated_at": now,
            "event_sequence": int(datetime.now(timezone.utc).timestamp()) + 1,
            "payload": {"status": "ARRIVED", "mission_milestone": "ARRIVED", "id": BID},
        },
        to=room,
    )
    print("EMITTED", room, "status=arrived milestone=ARRIVED updated_at", now)
    print("BOOKING_DB", b.status)
