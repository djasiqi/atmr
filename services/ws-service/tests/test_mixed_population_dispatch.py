"""Gate B2 : cohérence relay dispatch/chat (unitaire, sans socket réel)."""

import json

from dedup import EventDeduper
from event_contract import event_criticality
from rooms import company_room


def test_relay_payload_shape_for_company_room():
    company_id = 42
    event_id = "evt-dispatch-1"
    room = company_room(company_id)
    body = {
        "room": room,
        "event_type": "booking_updated",
        "payload": {"event_id": event_id, "booking_id": 99},
        "criticality": "critical",
    }
    raw = json.dumps(body)
    parsed = json.loads(raw)
    assert parsed["room"] == "company_42"
    assert parsed["event_type"] == "booking_updated"
    assert event_criticality(parsed["event_type"]) == "critical"


def test_mixed_pop_dedup_same_event_two_clients():
    """Client B (ws-service) ne doit pas voir deux fois le même event_id."""
    d = EventDeduper()
    user_b = "user-canary-1"
    room = company_room(1)
    eid = "shared-event-1"
    assert d.should_emit(user_id=user_b, room=room, event_id=eid) is True
    assert d.should_emit(user_id=user_b, room=room, event_id=eid) is False
