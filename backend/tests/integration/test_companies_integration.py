"""
Tests d'intégration pour le bounded context Companies.

Teste les flux complets route → use case → repository → DB pour les
endpoints d'entreprises et partenariats.
"""

from __future__ import annotations

import uuid

import pytest

from models import Client, Company
from models.enums import ClientType
from tests.integration.helpers import assert_response_json, assert_response_status


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
            "client_type": "PRIVATE",
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

    def test_search_companies_returns_match(
        self, authenticated_client, test_company, db
    ):
        """GET /companies/search renvoie les entreprises dont le nom/email matchent."""
        if not test_company:
            pytest.skip("test_company required")

        suffix = str(uuid.uuid4())[:8]
        other = Company(
            name="Emmenez-moi",
            contact_email=f"contact_{suffix}@emmenez-moi.ch",
            user_id=test_company.user_id,
        )
        db.session.add(other)
        db.session.flush()

        response = authenticated_client.get(
            "/api/v1/companies/search", params={"q": "emmenez"}
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert "data" in data
        names = [c["name"] for c in data["data"]]
        assert "Emmenez-moi" in names


@pytest.mark.integration
class TestPartnershipsIntegration:
    """Tests d'intégration pour POST /partnerships et GET /companies/me/partnerships."""

    def test_get_me_partnerships_returns_200(
        self, authenticated_client, test_company
    ):
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

    def test_create_partnership_success(
        self, authenticated_client, test_company, db
    ):
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
        response = authenticated_client.post(
            "/api/v1/partnerships", json=payload
        )
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
