from __future__ import annotations

from celery import shared_task  # pyright: ignore[reportMissingImports]


@shared_task(name="events.handle_domain_event")
def handle_domain_event(event: dict) -> None:  # type: ignore[type-arg]
    """Entrée Celery pour dispatcher un DomainEvent vers ses handlers."""
    from services.events.handlers_registry import dispatch_event

    dispatch_event(event)

