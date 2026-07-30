"""Lot 1 P0 — sécurité favoris geocode (multi-tenant + JWT)."""

from __future__ import annotations

import uuid

import pytest
from flask_jwt_extended import create_access_token

from models import Company, User, UserRole
from models.medical import FavoritePlace


def _company_headers(client, user, company_id: int | None = None):
    claims = {
        "role": user.role.value,
        "company_id": company_id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def company_a(db, sample_user):
    existing = Company.query.filter_by(user_id=sample_user.id).first()
    if existing:
        return existing
    company = Company()
    company.name = "Entreprise A Favoris"
    company.user_id = sample_user.id
    company.address = "Rue A 1"
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    db.session.refresh(company)
    return company


@pytest.fixture
def company_b_user(db):
    uid = str(uuid.uuid4())[:8]
    user = User()
    user.username = f"company_b_{uid}"
    user.email = f"company-b-{uid}@test.ch"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    company = Company()
    company.name = "Entreprise B Favoris"
    company.user_id = user.id
    company.address = "Rue B 1"
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    db.session.refresh(user)
    db.session.refresh(company)
    return user, company


@pytest.fixture
def favorite_a(db, company_a):
    fav = FavoritePlace()
    fav.company_id = company_a.id
    fav.label = "Clinique Alpha"
    fav.address = "Rue Alpha 10, Genève"
    fav.lat = 46.2
    fav.lon = 6.1
    db.session.add(fav)
    db.session.flush()
    return fav


@pytest.fixture
def favorite_b(db, company_b_user):
    _user, company = company_b_user
    fav = FavoritePlace()
    fav.company_id = company.id
    fav.label = "Clinique Beta Secret"
    fav.address = "Rue Beta 20, Lausanne"
    fav.lat = 46.5
    fav.lon = 6.6
    db.session.add(fav)
    db.session.flush()
    return fav


class TestGeocodeFavoritesSecurity:
    def test_public_autocomplete_ignores_company_id_no_favorites(
        self, client, company_a, favorite_a, favorite_b
    ):
        """Anonyme + company_id → aucun favori (endpoint public)."""
        response = client.get(
            f"/api/v1/geocode/autocomplete?q=Clinique&company_id={company_a.id}&limit=8"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert all(item.get("source") != "favorite" for item in data)
        assert not any("Beta Secret" in (item.get("label") or "") for item in data)
        assert not any("Alpha" in (item.get("label") or "") for item in data)

    def test_favorites_anonymous_returns_401(self, client, favorite_a):
        response = client.get("/api/v1/geocode/favorites/autocomplete?q=Clinique")
        assert response.status_code == 401

    def test_company_a_cannot_see_company_b_favorites(
        self, client, sample_user, company_a, favorite_a, favorite_b
    ):
        headers = _company_headers(client, sample_user, company_id=company_a.id)
        response = client.get(
            "/api/v1/geocode/favorites/autocomplete?q=Clinique",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        labels = [item.get("label") or "" for item in data]
        assert any("Alpha" in label for label in labels)
        assert not any("Beta Secret" in label for label in labels)
        assert response.headers.get("Cache-Control") == "private, no-store"

    def test_jwt_a_with_company_id_b_ignored(
        self, client, sample_user, company_a, company_b_user, favorite_a, favorite_b
    ):
        """JWT entreprise A + query company_id=B → favoris de A uniquement (B ignoré)."""
        _user_b, company_b = company_b_user
        headers = _company_headers(client, sample_user, company_id=company_a.id)
        response = client.get(
            f"/api/v1/geocode/favorites/autocomplete?q=Clinique&company_id={company_b.id}",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        labels = [item.get("label") or "" for item in data]
        assert any("Alpha" in label for label in labels)
        assert not any("Beta Secret" in label for label in labels)

    def test_log_correlation_opaque_no_raw_address(self):
        from routes.geocode import _geocode_log_correlation

        raw = "Rue Gabrielle-Perret-Gentil 4, 1205 Genève"
        corr = _geocode_log_correlation(raw)
        assert corr
        assert raw not in corr
        assert "Gabrielle" not in corr
        assert len(corr) == 16
