from dedup import EventDeduper


def test_dedup_same_key():
    d = EventDeduper()
    assert d.should_emit(user_id="u1", room="company_1", event_id="e1") is True
    assert d.should_emit(user_id="u1", room="company_1", event_id="e1") is False


def test_dedup_reconnect_new_sid_same_user():
    d = EventDeduper()
    assert d.should_emit(user_id="uuid-abc", room="company_2", event_id="e2") is True
    assert d.should_emit(user_id="uuid-abc", room="company_2", event_id="e2") is False
