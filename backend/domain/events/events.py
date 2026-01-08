from __future__ import annotations

from dataclasses import dataclass

from domain.events.base import DomainEvent


@dataclass(frozen=True, slots=True)
class BookingCreatedEvent(DomainEvent):
    event_type = "BookingCreatedEvent"

    booking_id: int = 0
    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class BookingAssignedEvent(DomainEvent):
    event_type = "BookingAssignedEvent"

    booking_id: int = 0
    company_id: int | None = None
    driver_id: int | None = None


@dataclass(frozen=True, slots=True)
class DispatchRunCompletedEvent(DomainEvent):
    event_type = "DispatchRunCompletedEvent"

    company_id: int = 0
    dispatch_run_id: int = 0
    assignments_count: int = 0
    date_str: str | None = None


@dataclass(frozen=True, slots=True)
class DriverLocationUpdatedEvent(DomainEvent):
    event_type = "DriverLocationUpdatedEvent"

    driver_id: int = 0
    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class DispatchRequestedEvent(DomainEvent):
    """Event utilitaire pour découpler les triggers dispatch
    (booking/driver changes)."""

    event_type = "DispatchRequestedEvent"

    company_id: int = 0
    action: str = "update"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BookingUpdatedEvent(DomainEvent):
    """Événement déclenché quand un booking est mis à jour."""

    event_type = "BookingUpdatedEvent"

    booking_id: int = 0
    driver_id: int | None = None
    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class BookingCancelledEvent(DomainEvent):
    """Événement déclenché quand un booking est annulé."""

    event_type = "BookingCancelledEvent"

    booking_id: int = 0
    driver_id: int | None = None
    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class DriverNewBookingEvent(DomainEvent):
    """Événement déclenché quand un nouveau booking est assigné à un driver."""

    event_type = "DriverNewBookingEvent"

    booking_id: int = 0
    driver_id: int = 0
    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class AssignmentCancelledEvent(DomainEvent):
    """Événement déclenché quand un assignment est annulé (release d'un booking)."""

    event_type = "AssignmentCancelledEvent"

    assignment_id: str = ""
    booking_id: int = 0
    driver_id: int = 0
    company_id: int = 0
