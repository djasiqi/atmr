from delivery_metrics import record_ack, record_delivery_attempt, stats


def test_delivery_attempt_critical_only():
    record_delivery_attempt("booking_updated", "e1", "u1", "company_1")
    record_delivery_attempt("driver_location_update", "e2", "u1", "company_1")
    s = stats()
    assert s["delivery_attempts_critical"] == 1
    record_ack("e1")
    assert stats()["event_acks_received"] == 1
