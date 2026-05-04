from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app import create_app
from models import Booking, Invoice, TransportRequest


def run_demo_sanity_check() -> int:
    app = create_app()
    with app.app_context():
        today = datetime.now(UTC).date()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)

        booking_count = Booking.query.count()
        request_count = TransportRequest.query.count()
        invoice_count = Invoice.query.count()

        today_bookings = Booking.query.filter(
            Booking.scheduled_time >= datetime.combine(today, datetime.min.time())
        ).count()
        yesterday_bookings = Booking.query.filter(
            Booking.scheduled_time >= datetime.combine(yesterday, datetime.min.time()),
            Booking.scheduled_time < datetime.combine(today, datetime.min.time()),
        ).count()
        tomorrow_bookings = Booking.query.filter(
            Booking.scheduled_time >= datetime.combine(tomorrow, datetime.min.time()),
        ).count()

        print(
            f"[demo-sanity] bookings={booking_count} requests={request_count} invoices={invoice_count}"
        )
        print(
            f"[demo-sanity] yesterday={yesterday_bookings} today={today_bookings} tomorrow={tomorrow_bookings}"
        )

        if booking_count == 0 or request_count == 0:
            print("[demo-sanity] ERROR: dataset démo vide.")
            return 1
        if today_bookings == 0:
            print("[demo-sanity] ERROR: aucun transport aujourd'hui.")
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(run_demo_sanity_check())
