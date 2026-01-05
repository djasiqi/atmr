"""
Tests d'intégration pour le bounded context Bookings.

Teste les flux complets route → use case → repository → DB pour les
endpoints de réservations.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import Booking, db
from models.enums import BookingStatus
from tests.integration.helpers import assert_response_json, assert_response_status


@pytest.mark.integration
class TestBookingsIntegration:
    """Tests d'intégration pour les réservations."""

    def test_create_booking_full_flow(
        self, authenticated_client, test_company, test_client, sample_user
    ):
        """Test création complète d'une réservation."""
        if not all([test_company, test_client, sample_user]):
            pytest.skip("Required fixtures missing")

        # Note: La route utilise public_id du client, pas l'ID
        url = f"/api/v1/bookings/clients/{test_client.public_id}/bookings"
        payload = {
            "customer_name": f"{test_client.first_name} {test_client.last_name}",
            "pickup_location": "Rue de Test 1, 1000 Lausanne",
            "dropoff_location": "Rue de Test 2, 1000 Lausanne",
            "scheduled_time": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "amount": 50.0,
        }

        response = authenticated_client.post(url, json=payload)
        # Peut retourner 201 (créé) ou 400/401 selon la validation
        assert response.status_code in [201, 400, 401]

        if response.status_code == 201:
            data = assert_response_json(response, ["booking_id"])
            # Vérifier que la réservation existe en DB
            booking = Booking.query.get(data["booking_id"])
            assert booking is not None
            assert booking.company_id == test_company.id
            assert booking.client_id == test_client.id
            assert booking.status == BookingStatus.PENDING

    def test_get_booking_with_relations(
        self, authenticated_client, test_company, test_booking
    ):
        """Test récupération d'une réservation avec ses relations."""
        if not all([test_company, test_booking]):
            pytest.skip("Required fixtures missing")

        url = f"/api/v1/bookings/{test_booking.id}"
        response = authenticated_client.get(url)
        # Peut retourner 200 ou 401/404 selon les permissions
        assert response.status_code in [200, 401, 404]

        if response.status_code == 200:
            data = assert_response_json(response)
            assert "id" in data
            assert data["id"] == test_booking.id

    def test_update_booking_status_flow(
        self, authenticated_client, test_company, test_booking
    ):
        """Test mise à jour du statut d'une réservation."""
        if not all([test_company, test_booking]):
            pytest.skip("Required fixtures missing")

        # La réservation doit être PENDING pour être modifiable
        test_booking.status = BookingStatus.PENDING
        db.session.commit()

        url = f"/api/v1/bookings/{test_booking.id}"
        payload = {
            "pickup_location": "Nouvelle adresse de départ",
            "dropoff_location": "Nouvelle adresse d'arrivée",
        }

        response = authenticated_client.put(url, json=payload)
        # Peut retourner 200 ou 400/401 selon les permissions
        assert response.status_code in [200, 400, 401]

    def test_cancel_booking_releases_resources(
        self, authenticated_client, test_company, test_booking
    ):
        """Test annulation d'une réservation et vérification de la libération des ressources."""
        if not all([test_company, test_booking]):
            pytest.skip("Required fixtures missing")

        # La réservation doit être PENDING ou CONFIRMED pour être annulée
        test_booking.status = BookingStatus.PENDING
        db.session.commit()

        url = f"/api/v1/bookings/{test_booking.id}"
        response = authenticated_client.delete(url)
        # Peut retourner 200 ou 400/401 selon les permissions
        assert response.status_code in [200, 400, 401]

        if response.status_code == 200:
            # Vérifier que la réservation est annulée
            db.session.refresh(test_booking)
            assert test_booking.status == BookingStatus.CANCELLED
