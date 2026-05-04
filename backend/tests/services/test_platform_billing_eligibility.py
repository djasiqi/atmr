"""Tests éligibilité commission plateforme."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from models.enums import BookingStatus
from services.platform_billing.eligibility import is_commissionable_platform


def _mock_booking(
    *,
    status=BookingStatus.COMPLETED,
    completed_at=None,
):
    b = MagicMock()
    b.status = status
    b.completed_at = completed_at or datetime.now(UTC)
    b.id = 1
    b.user_id = None
    b.client = None
    return b


def test_commissionable_requires_eligible_payload():
    b = _mock_booking()
    pl = {
        "source_code": "institution_request",
        "qualification": {"state": "eligible"},
        "observed_transport_amount": 45.0,
    }
    assert is_commissionable_platform(b, pl) is True


def test_not_commissionable_ambiguous():
    b = _mock_booking()
    pl = {
        "source_code": "institution_request",
        "qualification": {"state": "ambiguous"},
        "observed_transport_amount": 45.0,
    }
    assert is_commissionable_platform(b, pl) is False


def test_not_commissionable_without_completed_at():
    b = _mock_booking(completed_at=None)
    b.completed_at = None
    pl = {
        "source_code": "institution_request",
        "qualification": {"state": "eligible"},
        "observed_transport_amount": 45.0,
    }
    assert is_commissionable_platform(b, pl) is False


def test_not_commissionable_needs_review():
    b = _mock_booking()
    pl = {
        "source_code": "institution_request",
        "qualification": {"state": "needs_review"},
        "observed_transport_amount": 45.0,
    }
    assert is_commissionable_platform(b, pl) is False


def test_not_commissionable_excluded():
    b = _mock_booking()
    pl = {
        "source_code": "institution_request",
        "qualification": {"state": "excluded"},
        "observed_transport_amount": 45.0,
    }
    assert is_commissionable_platform(b, pl) is False


def test_not_commissionable_non_institution_source():
    b = _mock_booking()
    pl = {
        "source_code": "direct_client",
        "qualification": {"state": "eligible"},
        "observed_transport_amount": 45.0,
    }
    assert is_commissionable_platform(b, pl) is False


def test_not_commissionable_zero_amount():
    b = _mock_booking()
    pl = {
        "source_code": "institution_request",
        "qualification": {"state": "eligible"},
        "observed_transport_amount": 0,
    }
    assert is_commissionable_platform(b, pl) is False
