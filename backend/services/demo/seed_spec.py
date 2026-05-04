from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from models.enums import BookingStatus


@dataclass(frozen=True)
class DemoSeedProfile:
    companies: int
    institutions: int
    drivers: int
    vehicles: int
    patients: int
    transports_today: int
    transports_completed: int
    transports_pending: int
    transports_tomorrow: int
    invoices_draft: int
    invoices_sent: int
    invoices_paid: int


PROFILES: dict[str, DemoSeedProfile] = {
    "tiny": DemoSeedProfile(
        companies=1,
        institutions=1,
        drivers=2,
        vehicles=2,
        patients=8,
        transports_today=2,
        transports_completed=2,
        transports_pending=1,
        transports_tomorrow=0,
        invoices_draft=1,
        invoices_sent=1,
        invoices_paid=0,
    ),
    "sales": DemoSeedProfile(
        companies=3,
        institutions=3,
        drivers=8,
        vehicles=6,
        patients=30,
        transports_today=10,
        transports_completed=8,
        transports_pending=5,
        transports_tomorrow=5,
        invoices_draft=4,
        invoices_sent=3,
        invoices_paid=2,
    ),
}


def build_relative_transport_slots(
    reference_day: date,
    profile: DemoSeedProfile,
) -> list[tuple[datetime, BookingStatus]]:
    slots: list[tuple[datetime, BookingStatus]] = []

    for idx in range(profile.transports_completed):
        dt = datetime.combine(
            reference_day - timedelta(days=1),
            datetime.min.time(),
        ).replace(hour=8 + (idx % 10), minute=(idx * 7) % 60, tzinfo=UTC)
        slots.append((dt, BookingStatus.COMPLETED))

    for idx in range(profile.transports_today):
        dt = datetime.combine(reference_day, datetime.min.time()).replace(
            hour=7 + (idx % 11), minute=(idx * 6) % 60, tzinfo=UTC
        )
        status = (
            BookingStatus.ASSIGNED
            if idx < profile.transports_pending
            else BookingStatus.ACCEPTED
        )
        slots.append((dt, status))

    for idx in range(profile.transports_tomorrow):
        dt = datetime.combine(
            reference_day + timedelta(days=1), datetime.min.time()
        ).replace(hour=9 + (idx % 8), minute=(idx * 9) % 60, tzinfo=UTC)
        slots.append((dt, BookingStatus.PENDING))

    return slots
