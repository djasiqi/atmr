from unittest.mock import MagicMock, patch

import pytest

from services.worldline import reconciliation as rec


def test_list_stale_rejects_invalid_age():
    with pytest.raises(ValueError, match="min_age_minutes"):
        rec.list_stale_worldline_pending_payments(min_age_minutes=0)


def test_summarize_empty_when_query_returns_no_rows():
    mock_session = MagicMock()
    mock_q = MagicMock()
    mock_session.query.return_value = mock_q
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.limit.return_value = mock_q
    mock_q.all.return_value = []

    with patch.object(rec, "db") as mdb:
        mdb.session = mock_session
        summary = rec.summarize_stale_worldline_pending()
    assert summary["count"] == 0
    assert summary["payment_ids"] == []
    assert summary["items"] == []


def test_summarize_maps_row_fields():
    row = MagicMock()
    row.id = 7
    row.booking_id = 42
    ts = MagicMock()
    ts.isoformat.return_value = "2026-01-01T12:00:00+00:00"
    row.updated_at = ts
    row.worldline_hosted_checkout_id = "hc_abc"

    mock_session = MagicMock()
    mock_q = MagicMock()
    mock_session.query.return_value = mock_q
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.limit.return_value = mock_q
    mock_q.all.return_value = [row]

    with patch.object(rec, "db") as mdb:
        mdb.session = mock_session
        summary = rec.summarize_stale_worldline_pending()

    assert summary["count"] == 1
    assert summary["payment_ids"] == [7]
    assert summary["items"][0]["booking_id"] == 42
    assert summary["items"][0]["hosted_checkout_id"] == "hc_abc"
