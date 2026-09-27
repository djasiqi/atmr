"""Contrat des notifications opérationnelles chauffeur."""

from services.notifications.driver_operational_events import (
    EVENT_BOOKING_CANCELLED,
    EVENT_BOOKING_CHANGED,
    EVENT_DRIVER_ASSIGNED,
    EVENT_DRIVER_UNASSIGNED,
    EVENT_ROUTE_CHANGED,
    EVENT_SCHEDULE_CHANGED,
    plan_driver_operational_notifications,
)


def _plan(**kwargs):
    defaults = {
        "booking_id": 46778,
        "mutation_id": "mut-1",
        "time_label": "14:30",
        "occurred_at": "2026-09-27T10:00:00+00:00",
    }
    defaults.update(kwargs)
    return plan_driver_operational_notifications(**defaults)


def test_null_to_driver_is_one_assignment() -> None:
    planned = _plan(before_driver_id=None, after_driver_id=18)
    assert len(planned) == 1
    assert planned[0].event_type == EVENT_DRIVER_ASSIGNED
    assert planned[0].target_driver_id == 18
    assert planned[0].title == "Nouveau transport assigné"
    assert "14:30" in planned[0].body
    assert "patient" not in planned[0].body.lower()


def test_reassignment_notifies_old_and_new_once() -> None:
    planned = _plan(before_driver_id=12, after_driver_id=18)
    assert [item.event_type for item in planned] == [
        EVENT_DRIVER_UNASSIGNED,
        EVENT_DRIVER_ASSIGNED,
    ]
    assert planned[0].target_driver_id == 12
    assert planned[0].title == "Transport réattribué"
    assert planned[0].body == "Ce transport a été réattribué à un autre chauffeur."
    assert planned[0].payload["reason"] == "reassigned"
    assert planned[1].target_driver_id == 18


def test_unassign_notifies_only_previous_driver() -> None:
    planned = _plan(before_driver_id=18, after_driver_id=None)
    assert len(planned) == 1
    assert planned[0].event_type == EVENT_DRIVER_UNASSIGNED
    assert planned[0].target_driver_id == 18
    assert planned[0].body == "Ce transport ne vous est plus assigné."
    assert planned[0].payload["reason"] == "unassigned"
    assert "réattribué" not in planned[0].body


def test_schedule_change_is_one_push() -> None:
    planned = _plan(
        before_driver_id=18,
        after_driver_id=18,
        changes={
            "scheduled_time": {
                "from": "2026-09-27T14:30:00",
                "to": "2026-09-27T15:00:00",
            }
        },
    )
    assert len(planned) == 1
    assert planned[0].event_type == EVENT_SCHEDULE_CHANGED
    assert planned[0].title == "Horaire modifié"
    assert "14:30" in planned[0].body
    assert "15:00" in planned[0].body


def test_route_change_is_one_push() -> None:
    planned = _plan(
        before_driver_id=18,
        after_driver_id=18,
        changes={"pickup_location": {"from": "A", "to": "B"}},
    )
    assert len(planned) == 1
    assert planned[0].event_type == EVENT_ROUTE_CHANGED
    assert planned[0].title == "Itinéraire modifié"


def test_route_and_schedule_are_coalesced() -> None:
    planned = _plan(
        before_driver_id=18,
        after_driver_id=18,
        changes={
            "pickup_location": {"from": "A", "to": "B"},
            "scheduled_time": {
                "from": "2026-09-27T14:30:00",
                "to": "2026-09-27T15:00:00",
            },
        },
    )
    assert len(planned) == 1
    assert planned[0].event_type == EVENT_BOOKING_CHANGED
    assert planned[0].title == "Transport modifié"


def test_cancellation_does_not_also_unassign() -> None:
    planned = _plan(
        before_driver_id=18,
        after_driver_id=None,
        cancelled=True,
    )
    assert len(planned) == 1
    assert planned[0].event_type == EVENT_BOOKING_CANCELLED
    assert planned[0].title == "Transport annulé"


def test_retry_keeps_the_same_event_id() -> None:
    first = _plan(before_driver_id=None, after_driver_id=18)
    second = _plan(before_driver_id=None, after_driver_id=18)
    assert first[0].event_id == second[0].event_id
    assert first[0].payload["dedupe_key"] == f"event:{first[0].event_id}"


def test_unassigned_booking_change_sends_nothing() -> None:
    planned = _plan(
        before_driver_id=None,
        after_driver_id=None,
        changes={"scheduled_time": {"from": "14:30", "to": "15:00"}},
    )
    assert planned == []


def test_payload_carries_business_fields() -> None:
    planned = _plan(
        before_driver_id=18,
        after_driver_id=18,
        mission_anchor_booking_id=100,
        changes={"dropoff_location": {"from": "A", "to": "B"}},
    )
    payload = planned[0].payload
    assert payload["booking_id"] == 46778
    assert payload["mission_anchor_booking_id"] == 100
    assert payload["event_type"] == EVENT_ROUTE_CHANGED
    assert payload["deep_link"] == "lirie://driver/bookings/46778"
    assert payload["occurred_at"] == "2026-09-27T10:00:00+00:00"
    assert "client_name" not in payload
