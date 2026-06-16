"""Tests unitaires pour l'expiration des offres institution."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from models import OfferStatus, RequestOffer


@patch("shared.time_utils.now_utc")
def test_is_expired_interprets_naive_expires_at_as_geneva(mock_now):
    """expires_at naïf doit être interprété en Europe/Zurich, pas en UTC."""
    mock_now.return_value = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)
    offer = RequestOffer()
    offer.status = OfferStatus.PENDING.value
    # 10:00 Genève (hiver) = 09:00 UTC → expiré à 09:30 UTC
    offer.expires_at = datetime(2026, 1, 15, 10, 0, 0)

    assert offer.is_expired is True


def test_is_expired_false_when_expires_at_in_future_utc():
    offer = RequestOffer()
    offer.status = OfferStatus.PENDING.value
    offer.expires_at = datetime.now(UTC) + timedelta(hours=2)

    assert offer.is_expired is False


def test_serialize_expires_at_as_utc_z_instant():
    """expires_at est un instant absolu, pas une heure murale mission."""
    offer = RequestOffer()
    offer.id = 1
    offer.transport_request_id = 10
    offer.company_id = 5
    offer.mode = "sequential"
    offer.order = 1
    offer.status = OfferStatus.PENDING.value
    offer.expires_at = datetime(2026, 6, 16, 10, 30, 0, tzinfo=UTC)

    data = offer.serialize
    assert data["expires_at"] == "2026-06-16T10:30:00Z"
    assert data["expires_at"].endswith("Z")
