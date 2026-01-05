from __future__ import annotations

from typing import Any

from application.events.event_bus import publish_event, set_event_bus
from domain.events.events import BookingCreatedEvent
from infrastructure.events.celery_event_bus import CeleryEventBus
from infrastructure.events.in_memory_event_bus import InMemoryEventBus
from services import event_handlers_registry as registry


def test_in_memory_event_bus_dispatches_to_registry(monkeypatch) -> None:
    # reset registry (tests ont droit à SLF001)
    registry._HANDLERS.clear()

    received: list[dict[str, Any]] = []

    def handler(evt: dict[str, Any]) -> None:
        received.append(evt)

    registry.register("BookingCreatedEvent", handler)
    set_event_bus(InMemoryEventBus())

    publish_event(BookingCreatedEvent(booking_id=123, company_id=42))

    assert len(received) == 1
    assert received[0]["event_type"] == "BookingCreatedEvent"
    assert received[0]["booking_id"] == 123
    assert received[0]["company_id"] == 42


def test_celery_event_bus_sends_task(monkeypatch) -> None:
    calls: list[tuple[str, list[Any]]] = []

    class _FakeCelery:
        def send_task(self, name: str, args: list[Any] | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
            calls.append((name, list(args or [])))

    import celery_app

    monkeypatch.setattr(celery_app, "celery", _FakeCelery())
    set_event_bus(CeleryEventBus())

    publish_event(BookingCreatedEvent(booking_id=1, company_id=2))

    assert calls
    assert calls[0][0] == "events.handle_domain_event"
    payload = calls[0][1][0]
    assert payload["event_type"] == "BookingCreatedEvent"


def test_registry_has_default_handlers_for_created_and_location_updated() -> None:
    # Vérifie que les handlers "par défaut" existent et ne crashent pas
    registry.dispatch_event(
        {"event_type": "BookingCreatedEvent", "booking_id": 1, "company_id": 2}
    )
    registry.dispatch_event(
        {"event_type": "DriverLocationUpdatedEvent", "driver_id": 1, "company_id": 2}
    )
