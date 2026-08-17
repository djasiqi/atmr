"""Couverture des packages ``application``."""

from __future__ import annotations

import application
from application import dispatch, events
from application.dispatch.dispatch_use_case import DispatchUseCase
from application.events.event_bus import (
    EventBus,
    get_event_bus,
    publish_event,
    set_event_bus,
)


def test_application_package_export():
    assert application.__all__ == []
    assert application.__doc__


def test_dispatch_package_export():
    assert dispatch.__all__ == ["DispatchUseCase"]
    assert dispatch.DispatchUseCase is DispatchUseCase


def test_events_package_export():
    assert events.EventBus is EventBus
    assert events.get_event_bus is get_event_bus
    assert events.publish_event is publish_event
    assert events.set_event_bus is set_event_bus
