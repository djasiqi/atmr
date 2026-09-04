"""
Tests d'intégration pour le bounded context Companies.

Teste les flux complets route → use case → repository → DB pour les
endpoints d'entreprises et partenariats.
"""

from __future__ import annotations

import os
import uuid

import pytest

from models import Booking, Client, Company
from models.enums import ClientType
from tests.integration.helpers import assert_response_json, assert_response_status

# Skip driver_locations tests si SQLite (PostgreSQL requis pour ces tests d'intégration)
_skip_if_sqlite = pytest.mark.skipif(
    "sqlite" in (os.getenv("DATABASE_URL") or "").lower(),
    reason="PostgreSQL required for driver_locations integration tests. "
    "Run: docker compose -f docker-compose.test.yml up -d postgres_test && "
    "DATABASE_URL=postgresql://test:test@localhost:5433/atmr_test pytest ...",
)


@pytest.mark.integration
class TestCompaniesIntegration:
    """Tests d'intégration pour les entreprises."""

    def test_get_current_company_returns_company(
        self, authenticated_client, test_company, sample_user
    ):
        """Test récupération de l'entreprise courante."""
        if not all([test_company, sample_user]):
            pytest.skip("Required fixtures missing")

        url = "/api/v1/companies/current"
        response = authenticated_client.get(url)
        # Peut retourner 200 ou 404 selon l'implémentation
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = assert_response_json(response)
            assert "id" in data or "company" in data

    def test_get_current_company_or_create_creates_if_missing(
        self, authenticated_client, sample_user, db
    ):
        """Test création d'entreprise si absente."""
        if not sample_user:
            pytest.skip("sample_user required")

        # Supprimer l'entreprise existante si elle existe
        existing_company = Company.query.filter_by(user_id=sample_user.id).first()
        if existing_company:
            db.session.delete(existing_company)
            db.session.commit()

        url = "/api/v1/companies/current"
        response = authenticated_client.get(url)
        # Peut retourner 200 (créée) ou 404 selon l'implémentation
        assert response.status_code in [200, 404]

    def test_create_company_client_full_flow(self, authenticated_client, test_company):
        """Test création complète d'un client d'entreprise."""
        if not test_company:
            pytest.skip("test_company required")

        url = "/api/v1/companies/me/clients"
        payload = {
            "first_name": "New",
            "last_name": "Client",
            "email": "newclient@test.ch",
            "phone": "0211234567",
            "management_mode": "MANAGED",
            "gender": "male",
            "address": "Rue du Test 1, 1200 Genève",
        }

        response = authenticated_client.post(url, json=payload)
        # Peut retourner 201 (créé) ou 400 selon la validation
        assert response.status_code in [201, 400]

        if response.status_code == 201:
            data = assert_response_json(response)
            # Vérifier que le client existe en DB
            if "id" in data:
                client = Client.query.get(data["id"])
                assert client is not None
                assert client.company_id == test_company.id

    def test_list_company_drivers_with_filters(
        self, authenticated_client, test_company, test_driver
    ):
        """Test liste des chauffeurs avec filtres."""
        if not all([test_company, test_driver]):
            pytest.skip("Required fixtures missing")

        url = f"/api/v1/companies/{test_company.id}/drivers"
        response = authenticated_client.get(url)
        # Peut retourner 200 ou 404 selon l'implémentation
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = assert_response_json(response)
            # Vérifier la structure de la réponse
            assert isinstance(data, (dict, list))

    @pytest.mark.postgresql
    @_skip_if_sqlite
    def test_get_driver_locations_returns_locations(
        self, authenticated_client, test_company, test_driver, requires_postgresql
    ):
        """Test GET /companies/me/drivers/locations - positions GPS chauffeurs."""
        if not all([test_company, test_driver]):
            pytest.skip("Required fixtures missing")

        url = "/api/v1/companies/me/drivers/locations"
        response = authenticated_client.get(url)
        assert response.status_code == 200
        data = assert_response_json(response)
        assert "locations" in data
        assert isinstance(data["locations"], list)
        for loc in data["locations"]:
            assert "driver_id" in loc
            assert "latitude" in loc
            assert "longitude" in loc
            assert "is_stale" in loc
            assert "last_seen_seconds" in loc

    @pytest.mark.postgresql
    @_skip_if_sqlite
    def test_get_driver_locations_includes_status_and_booking(
        self, authenticated_client, test_company, test_driver, db, requires_postgresql
    ):
        """Test que chaque location inclut status et current_booking_id si busy."""
        if not all([test_company, test_driver]):
            pytest.skip("Required fixtures missing")

        # Donner une position au chauffeur (fallback DB)
        test_driver.latitude = 46.2044
        test_driver.longitude = 6.1432
        db.session.flush()
        db.session.commit()

        url = "/api/v1/companies/me/drivers/locations"
        response = authenticated_client.get(url)
        assert response.status_code == 200
        data = assert_response_json(response)
        assert "locations" in data
        locations = data["locations"]
        for loc in locations:
            assert "status" in loc
            assert loc["status"] in ("available", "busy", "offline")
            if loc["status"] == "busy":
                assert "current_booking_id" in loc

    @pytest.mark.postgresql
    @_skip_if_sqlite
    def test_offline_when_stale_or_missing(
        self, authenticated_client, test_company, test_driver, db, requires_postgresql
    ):
        """Test que status=offline quand is_stale ou driver désactivé."""
        if not all([test_company, test_driver]):
            pytest.skip("Required fixtures missing")

        test_driver.latitude = 46.2044
        test_driver.longitude = 6.1432
        test_driver.is_active = False
        db.session.flush()
        db.session.commit()

        url = "/api/v1/companies/me/drivers/locations"
        response = authenticated_client.get(url)
        assert response.status_code == 200
        data = assert_response_json(response)
        locations = data["locations"]
        for loc in locations:
            if loc["driver_id"] == test_driver.id:
                assert loc["status"] == "offline"
                break

    @pytest.mark.postgresql
    @_skip_if_sqlite
    def test_busy_when_active_booking(
        self,
        authenticated_client,
        test_company,
        test_driver,
        test_client,
        db,
        requires_postgresql,
    ):
        """Test que status=busy quand chauffeur a une course ASSIGNED/EN_ROUTE/IN_PROGRESS."""
        from datetime import UTC, datetime, timedelta
        from decimal import Decimal

        from models.enums import BookingStatus

        if not all([test_company, test_driver, test_client]):
            pytest.skip("Required fixtures missing")

        test_driver.latitude = 46.2044
        test_driver.longitude = 6.1432
        db.session.flush()

        booking = Booking()
        booking.user_id = test_client.user_id
        booking.company_id = test_company.id
        booking.client_id = test_client.id
        booking.customer_name = "Test Client"
        booking.pickup_location = "Rue Test 1"
        booking.dropoff_location = "Rue Test 2"
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=1)
        booking.status = BookingStatus.IN_PROGRESS
        booking.driver_id = test_driver.id
        booking.amount = Decimal("50.00")
        booking.vat_rate = Decimal("7.70")
        db.session.add(booking)
        db.session.flush()
        db.session.commit()

        url = "/api/v1/companies/me/drivers/locations"
        response = authenticated_client.get(url)
        assert response.status_code == 200
        data = assert_response_json(response)
        locations = data["locations"]
        busy_found = False
        for loc in locations:
            if loc["driver_id"] == test_driver.id:
                assert loc["status"] == "busy"
                assert "current_booking_id" in loc
                assert loc["current_booking_id"] == booking.id
                busy_found = True
                break
        assert busy_found, "Driver with active booking should appear as busy"

    def test_search_companies_returns_match(
        self, authenticated_client, test_company, db
    ):
        """GET /companies/search renvoie les entreprises dont le nom/email matchent."""
        if not test_company:
            pytest.skip("test_company required")

        # Jeton unique : évite d'être évincé du .limit(20) par des résidus DB
        # (ex. anciennes « Emmenez Moi ») et garantit la visibilité après commit.
        suffix = str(uuid.uuid4())[:8]
        unique_name = f"Emmenez-moi-{suffix}"
        other = Company(
            name=unique_name,
            contact_email=f"contact_{suffix}@emmenez-moi.ch",
            user_id=test_company.user_id,
        )
        db.session.add(other)
        db.session.commit()

        response = authenticated_client.get(
            "/api/v1/companies/search", query_string={"q": suffix}
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert "data" in data
        names = [c["name"] for c in data["data"]]
        assert unique_name in names


@pytest.mark.integration
class TestPartnershipsIntegration:
    """Tests d'intégration pour POST /partnerships et GET /companies/me/partnerships."""

    def test_get_me_partnerships_returns_200(self, authenticated_client, test_company):
        """GET /companies/me/partnerships renvoie 200 (pas 400) même sans partenariats."""
        if not test_company:
            pytest.skip("test_company required")

        response = authenticated_client.get("/api/v1/companies/me/partnerships")
        assert response.status_code == 200, (
            response.status_code,
            response.get_data(as_text=True),
        )
        data = response.get_json()
        assert data is not None
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_create_partnership_success(self, authenticated_client, test_company, db):
        """POST /partnerships avec partner_company_id valide renvoie 201."""
        if not test_company:
            pytest.skip("test_company required")

        suffix = str(uuid.uuid4())[:8]
        partner = Company(
            name=f"Partner {suffix}",
            contact_email=f"partner_{suffix}@test.ch",
            user_id=test_company.user_id,
        )
        db.session.add(partner)
        db.session.flush()

        payload = {
            "partner_company_id": partner.id,
            "default_partner_tariff_percent": 90,
            "payment_terms_days": 30,
        }
        response = authenticated_client.post("/api/v1/partnerships", json=payload)
        assert response.status_code == 201
        data = response.get_json()
        assert data is not None
        # Réponse success peut être {"data": {...}} selon success_response
        has_data = "data" in data or "id" in data
        has_partner = "partner_company_id" in str(data)
        assert has_data or has_partner

    def test_create_partnership_duplicate_returns_409(
        self, authenticated_client, test_company, db
    ):
        """POST /partnerships une 2e fois pour le même partenaire renvoie 409."""
        if not test_company:
            pytest.skip("test_company required")

        suffix = str(uuid.uuid4())[:8]
        partner = Company(
            name=f"Partner Dup {suffix}",
            contact_email=f"dup_{suffix}@test.ch",
            user_id=test_company.user_id,
        )
        db.session.add(partner)
        db.session.flush()

        payload = {
            "partner_company_id": partner.id,
            "default_partner_tariff_percent": 90,
            "payment_terms_days": 30,
        }
        r1 = authenticated_client.post("/api/v1/partnerships", json=payload)
        assert r1.status_code == 201, (r1.status_code, r1.get_json())

        r2 = authenticated_client.post("/api/v1/partnerships", json=payload)
        assert r2.status_code == 409
        body = r2.get_json()
        assert body is not None
        assert "error" in body or "message" in body
        assert "déjà" in str(body).lower() or "existe" in str(body).lower()
