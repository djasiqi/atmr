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
    # ✅ Permet d'éviter les "self-notifications" (ex: le chauffeur ne doit pas
    # recevoir une notif pour un statut qu'il vient lui-même de changer).
    actor_role: str | None = None  # "driver" | "company" | "system" | None
    actor_id: int | None = None
    # P0.2: Source stable pour fallback quand actor_role manquant (job, import, migration).
    source: str | None = None  # "driver_api" | "company_api" | "system"
    # Diff partiel pour générer des notifications "pro" basées sur ce qui a changé.
    # Exemple:
    # {
    #   "scheduled_time": {"from": "2026-01-20T14:20:00", "to": "2026-01-20T14:35:00"},
    #   "pickup_location": {"from": "...", "to": "..."},
    #   "dropoff_location": {"from": "...", "to": "..."},
    #   "notes": {"from": "...", "to": "..."},
    # }
    changes: dict[str, dict[str, str | None]] | None = None


@dataclass(frozen=True, slots=True)
class BookingCancelledEvent(DomainEvent):
    """Événement déclenché quand un booking est annulé."""

    event_type = "BookingCancelledEvent"

    booking_id: int = 0
    driver_id: int | None = None
    company_id: int | None = None
    actor_role: str | None = None  # "driver" | "company" | "institution" | "system"
    actor_id: int | None = None  # domain ID (driver_id, company_id, institution_id)
    cancel_reason: str | None = None
    cancel_source: str | None = (
        None  # "company_api" | "driver_api" | "institution_api" | "system"
    )


@dataclass(frozen=True, slots=True)
class DriverNewBookingEvent(DomainEvent):
    """Événement déclenché quand un nouveau booking est assigné à un driver."""

    event_type = "DriverNewBookingEvent"

    booking_id: int = 0
    driver_id: int = 0
    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class DriverBookingReassignedEvent(DomainEvent):
    """Événement déclenché quand une mission est retirée à un chauffeur (réassignée)."""

    event_type = "DriverBookingReassignedEvent"

    booking_id: int = 0
    old_driver_id: int = 0
    new_driver_id: int | None = None
    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class AssignmentCancelledEvent(DomainEvent):
    """Événement déclenché quand un assignment est annulé (release d'un booking)."""

    event_type = "AssignmentCancelledEvent"

    assignment_id: str = ""
    booking_id: int = 0
    driver_id: int = 0
    company_id: int = 0
