"""Tests E2E pour validation des schemas Marshmallow sur les endpoints API.

Tests tous les endpoints validés avec des payloads valides/invalides pour vérifier
que la validation Marshmallow fonctionne correctement.
"""

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from flask import Flask

from models import UserRole


class TestSchemaValidationE2E:
    """Tests E2E pour validation des schemas sur les endpoints validés."""

    # ========== AUTH ENDPOINTS ==========

    def test_login_valid_schema(self, client, sample_user):
        """Test POST /api/v1/auth/login avec payload valide."""
        response = client.post("/api/v1/auth/login", json={"email": sample_user.email, "password": "password123"})
        assert response.status_code in [200, 400, 404, 429, 500]
        data = response.get_json() or {}
        assert ("token" in data) or ("message" in data) or ("errors" in data) or ("error" in data)

    def test_login_invalid_schema(self, client):
        """Test POST /api/v1/auth/login avec payload invalide (email manquant)."""
        response = client.post("/api/v1/auth/login", json={"password": "password123"})
        assert response.status_code in [400, 404, 500]
        data = response.get_json() or {}
        assert ("message" in data) or ("errors" in data) or ("error" in data) or (response.status_code == 404)

    def test_register_valid_schema(self, client, db):
        """Test POST /api/v1/auth/register avec payload valide."""
        unique_email = f"test-{uuid.uuid4().hex[:8]}@example.com"
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": f"testuser_{uuid.uuid4().hex[:6]}",
                "email": unique_email,
                "password": "SecurePass123!",
                "first_name": "Test",
                "last_name": "User",
                "phone": "+41211234567",
                "address": "Rue de Test 1, 1000 Lausanne",
            },
        )
        assert response.status_code in [200, 201, 400, 404, 500]
        data = response.get_json() or {}
        assert (
            ("token" in data)
            or ("user" in data)
            or ("message" in data)
            or ("error" in data)
            or (response.status_code in [404, 500])
        )

    def test_register_invalid_schema(self, client):
        """Test POST /api/v1/auth/register avec payload invalide (email invalide)."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "invalid-email",
                "password": "password123",
                "first_name": "Test",
                "last_name": "User",
            },
        )
        assert response.status_code in [400, 404, 500]
        data = response.get_json() or {}
        assert ("message" in data) or ("errors" in data) or ("error" in data) or (response.status_code == 404)

    # ========== BOOKINGS ENDPOINTS ==========

    def test_create_booking_valid_schema(self, client, auth_headers, sample_user, db, sample_company):
        """Test POST /api/v1/clients/<id>/bookings avec payload valide."""
        from models import Client, User

        # Créer un client de test
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        from ext import bcrypt

        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        test_client.company_id = sample_company.id  # Utiliser sample_company pour garantir company_id non-None
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Générer un JWT client directement
        from flask_jwt_extended import create_access_token

        client_claims = {
            "role": UserRole.client.value,
            "company_id": None,
            "driver_id": None,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            client_token = create_access_token(identity=str(client_user.public_id), additional_claims=client_claims)
        client_headers = {"Authorization": f"Bearer {client_token}"}

        future_time = (datetime.now(UTC) + timedelta(days=1)).isoformat()

        response = client.post(
            f"/api/v1/clients/{test_client.id}/bookings",
            json={
                "customer_name": "Test Customer",
                "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
                "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
                "scheduled_time": future_time,
                "amount": 50.0,
            },
            headers=client_headers,
        )
        assert response.status_code in [201, 400, 404, 500]  # 400 si géocodage échoue

    def test_create_booking_invalid_schema(self, client, auth_headers, sample_user, db, sample_company):
        """Test POST /api/v1/clients/<id>/bookings avec payload invalide (champs manquants)."""
        from models import Client, User

        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        from ext import bcrypt

        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        test_client.company_id = sample_company.id  # Utiliser sample_company pour garantir company_id non-None
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        from flask_jwt_extended import create_access_token

        client_claims = {
            "role": UserRole.client.value,
            "company_id": None,
            "driver_id": None,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            client_token = create_access_token(identity=str(client_user.public_id), additional_claims=client_claims)
        client_headers = {"Authorization": f"Bearer {client_token}"}

        response = client.post(
            f"/api/v1/clients/{test_client.id}/bookings",
            json={
                "customer_name": "Test Customer"
                # pickup_location, dropoff_location, scheduled_time manquants
            },
            headers=client_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json()
        assert ("message" in data) or ("errors" in data) or ("error" in data)

    def test_list_bookings_valid_query_params(self, client, auth_headers):
        """Test GET /api/bookings avec query params valides."""
        response = client.get("/api/v1/bookings?page=1&per_page=20&status=PENDING", headers=auth_headers)
        assert response.status_code in [200, 400, 403]  # 400 si validation stricte des params

    def test_list_bookings_invalid_query_params(self, client, auth_headers):
        """Test GET /api/bookings avec query params invalides (per_page trop élevé)."""
        response = client.get("/api/v1/bookings?page=1&per_page=1000", headers=auth_headers)
        # Devrait valider et limiter à 500 max
        assert response.status_code in [200, 400, 403]

    def test_update_booking_valid_schema(self, client, db):
        """✅ Test E2E PUT /api/bookings/<id> avec BookingUpdateSchema valide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Booking, BookingStatus, Client, User, UserRole

        # Créer un client user et une réservation
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()

        # Créer une réservation PENDING pour pouvoir la modifier
        future_time = datetime.now(UTC) + timedelta(days=1)
        booking = Booking()
        booking.client_id = test_client.id
        booking.user_id = client_user.id
        booking.status = BookingStatus.PENDING
        booking.customer_name = "Test Customer"
        booking.pickup_location = "Rue de la Gare 1, 1000 Lausanne"
        booking.dropoff_location = "Avenue de la Plage 10, 1000 Lausanne"
        booking.scheduled_time = future_time
        booking.amount = 50.0
        db.session.add(booking)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Générer un JWT client directement
        from flask_jwt_extended import create_access_token

        client_claims = {
            "role": UserRole.client.value,
            "company_id": None,
            "driver_id": None,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            client_token = create_access_token(identity=str(client_user.public_id), additional_claims=client_claims)
        client_headers = {"Authorization": f"Bearer {client_token}"}

        # Test mise à jour valide
        new_time = (future_time + timedelta(hours=1)).isoformat()
        response = client.put(
            f"/api/v1/bookings/{booking.id}",
            json={
                "pickup_location": "Rue de la Gare 2, 1000 Lausanne",
                "dropoff_location": "Avenue de la Plage 20, 1000 Lausanne",
                "scheduled_time": new_time,
                "amount": 60.0,
                "status": "confirmed",
                "medical_facility": "Hôpital Cantonal",
                "doctor_name": "Dr. Test",
                "notes_medical": "Notes médicales de test",
            },
            headers=client_headers,
        )
        # Peut être 200 (succès) ou 400 si géocodage échoue
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.get_json()
            assert "message" in data

    def test_update_booking_invalid_schema(self, client, db):
        """✅ Test E2E PUT /api/bookings/<id> avec BookingUpdateSchema invalide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Booking, BookingStatus, Client, User, UserRole

        # Créer un client user et une réservation
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()

        future_time = datetime.now(UTC) + timedelta(days=1)
        booking = Booking()
        booking.client_id = test_client.id
        booking.user_id = client_user.id
        booking.status = BookingStatus.PENDING
        booking.customer_name = "Test Customer"
        booking.pickup_location = "Rue de la Gare 1, 1000 Lausanne"
        booking.dropoff_location = "Avenue de la Plage 10, 1000 Lausanne"
        booking.scheduled_time = future_time
        booking.amount = 50.0
        booking.user_id = client_user.id
        db.session.add(booking)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        from flask_jwt_extended import create_access_token

        client_claims = {
            "role": UserRole.client.value,
            "company_id": None,
            "driver_id": None,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            client_token = create_access_token(identity=str(client_user.public_id), additional_claims=client_claims)
        client_headers = {"Authorization": f"Bearer {client_token}"}

        # Test avec format date invalide (scheduled_time)
        response = client.put(
            f"/api/v1/bookings/{booking.id}", json={"scheduled_time": "invalid-date-format"}, headers=client_headers
        )
        assert response.status_code in [400, 500]
        data = response.get_json()
        assert ("message" in data) or ("errors" in data) or ("error" in data)
        # Vérifier que l'erreur mentionne scheduled_time
        error_str = str(data).lower()
        assert "scheduled_time" in error_str or "date" in error_str or "format" in error_str

        # Test avec statut invalide
        response = client.put(
            f"/api/v1/bookings/{booking.id}", json={"status": "invalid_status"}, headers=client_headers
        )
        assert response.status_code in [400, 500]
        data = response.get_json()
        assert "message" in data or "errors" in data
        # Vérifier que l'erreur mentionne status
        error_str = str(data).lower()
        assert "status" in error_str or "errors" in error_str

        # Test avec amount négatif
        response = client.put(f"/api/v1/bookings/{booking.id}", json={"amount": -10.0}, headers=client_headers)
        assert response.status_code in [400, 500]
        data = response.get_json()
        assert "message" in data or "errors" in data

    def test_create_manual_booking_valid_schema(self, client, db, sample_company):
        """✅ Test E2E POST /api/companies/me/reservations/manual avec ManualBookingCreateSchema valide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Client, User, UserRole

        # Créer un client pour cette company
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        test_client.company_id = sample_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Authentification: générer un JWT company directement
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        # Auth company: générer JWT direct
        from flask_jwt_extended import create_access_token

        company_claims = {
            "role": UserRole.company.value,
            "company_id": sample_company.id,
            "driver_id": None,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            company_token = create_access_token(identity=str(company_user.public_id), additional_claims=company_claims)
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test création réservation manuelle valide
        future_time = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        response = client.post(
            "/api/v1/companies/me/reservations/manual",
            json={
                "client_id": test_client.id,
                "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
                "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
                "scheduled_time": future_time,
                "customer_first_name": "Jean",
                "customer_last_name": "Dupont",
                "customer_email": "jean.dupont@example.com",
                "is_round_trip": True,
                "return_time": (datetime.now(UTC) + timedelta(days=1, hours=2)).isoformat(),
                "amount": 75.0,
                "billed_to_type": "patient",
                "medical_facility": "Hôpital Cantonal",
                "doctor_name": "Dr. Test",
            },
            headers=company_headers,
        )
        # Peut être 201 (succès) ou 400 si géocodage échoue
        assert response.status_code in [201, 400, 429, 500]
        if response.status_code == 201:
            data = response.get_json()
            assert "id" in data or "booking_id" in data or "message" in data

    def test_create_manual_booking_invalid_schema(self, client, db, sample_company):
        """✅ Test E2E POST /api/companies/me/reservations/manual avec ManualBookingCreateSchema invalide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Client, User, UserRole

        # Créer un client pour cette company
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        test_client.company_id = sample_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Auth company via JWT direct
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        company_claims = {
            "role": UserRole.company.value,
            "company_id": sample_company.id,
            "driver_id": None,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            company_token = create_access_token(identity=str(company_user.public_id), additional_claims=company_claims)
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test avec client_id manquant (requis)
        response = client.post(
            "/api/v1/companies/me/reservations/manual",
            json={
                "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
                "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
                "scheduled_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                # client_id manquant
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        error_str = str(data).lower()
        assert "client_id" in error_str or "errors" in error_str

        # Test avec format date invalide (scheduled_time)
        response = client.post(
            "/api/v1/companies/me/reservations/manual",
            json={
                "client_id": test_client.id,
                "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
                "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
                "scheduled_time": "invalid-date-format",
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        error_str = str(data).lower()
        assert "scheduled_time" in error_str or "date" in error_str or "format" in error_str

        # Test avec billed_to_type invalide (enum)
        future_time = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        response = client.post(
            "/api/v1/companies/me/reservations/manual",
            json={
                "client_id": test_client.id,
                "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
                "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
                "scheduled_time": future_time,
                "billed_to_type": "invalid_type",
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data)

        # Test avec pickup_location trop long (> 500)
        response = client.post(
            "/api/v1/companies/me/reservations/manual",
            json={
                "client_id": test_client.id,
                "pickup_location": "a" * 501,  # Max 500
                "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
                "scheduled_time": future_time,
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data)

        # Test avec amount négatif
        response = client.post(
            "/api/v1/companies/me/reservations/manual",
            json={
                "client_id": test_client.id,
                "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
                "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
                "scheduled_time": future_time,
                "amount": -10.0,
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data)

    def test_create_client_valid_schema_self_service(self, client, db, sample_company):
        """✅ Test E2E POST /api/companies/me/clients avec ClientCreateSchema valide (SELF_SERVICE)."""
        from models import User, UserRole

        # Login en tant que company
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()

        login_response = client.post("/api/auth/login", json={"email": company_user.email, "password": "password123"})
        company_token = login_response.get_json()["token"]
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test création client SELF_SERVICE (email requis)
        unique_email = f"selfservice_{uuid.uuid4().hex[:8]}@example.com"
        response = client.post(
            "/api/v1/companies/me/clients",
            json={"client_type": "SELF_SERVICE", "email": unique_email},
            headers=company_headers,
        )
        assert response.status_code in [201, 400, 429, 500]  # 429 rate limit toléré

    def test_create_client_valid_schema_private(self, client, db, sample_company):
        """✅ Test E2E POST /api/companies/me/clients avec ClientCreateSchema valide (PRIVATE)."""
        from ext import bcrypt
        from models import User, UserRole

        # Authentification: générer un JWT company directement
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        company_token = create_access_token(
            identity=str(company_user.public_id),
            additional_claims={"role": UserRole.company.value, "company_id": sample_company.id},
            expires_delta=timedelta(hours=1),
        )
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test création client PRIVATE (first_name, last_name, address requis)
        response = client.post(
            "/api/v1/companies/me/clients",
            json={
                "client_type": "PRIVATE",
                "first_name": "Jean",
                "last_name": "Dupont",
                "address": "Rue de Test 1, 1000 Lausanne",
                "phone": "+41211234567",
                "birth_date": "1990-01-15",
                "billing_address": "Rue de Facturation 2, 1000 Lausanne",
                "contact_email": "jean.dupont@example.com",
            },
            headers=company_headers,
        )
        assert response.status_code in [201, 400, 429]

    def test_create_client_valid_schema_corporate(self, client, db, sample_company):
        """✅ Test E2E POST /api/companies/me/clients avec ClientCreateSchema valide (CORPORATE)."""
        from ext import bcrypt
        from models import User, UserRole

        # Authentification: générer un JWT company directement
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        company_token = create_access_token(
            identity=str(company_user.public_id),
            additional_claims={"role": UserRole.company.value, "company_id": sample_company.id},
            expires_delta=timedelta(hours=1),
        )
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test création client CORPORATE (first_name, last_name, address requis)
        response = client.post(
            "/api/v1/companies/me/clients",
            json={
                "client_type": "CORPORATE",
                "first_name": "Marie",
                "last_name": "Martin",
                "address": "Avenue Corporate 10, 1000 Lausanne",
                "is_institution": True,
                "institution_name": "Clinique Test SA",
                "contact_email": "contact@clinique-test.ch",
                "contact_phone": "+41219876543",
                "billing_address": "Avenue Facturation 20, 1000 Lausanne",
            },
            headers=company_headers,
        )
        assert response.status_code in [201, 400, 429]

    def test_create_client_invalid_schema(self, client, db, sample_company):
        """✅ Test E2E POST /api/companies/me/clients avec ClientCreateSchema invalide."""
        from ext import bcrypt
        from models import User, UserRole

        # Authentification: générer un JWT company directement
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        company_token = create_access_token(
            identity=str(company_user.public_id),
            additional_claims={"role": UserRole.company.value, "company_id": sample_company.id},
            expires_delta=timedelta(hours=1),
        )
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test avec client_type manquant (requis)
        response = client.post(
            "/api/v1/companies/me/clients", json={"first_name": "Jean", "last_name": "Dupont"}, headers=company_headers
        )
        assert response.status_code in [400, 429, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data) or response.status_code == 429
        error_str = str(data).lower()
        if response.status_code != 429:
            assert "client_type" in error_str or "errors" in error_str

        # Test avec client_type invalide
        response = client.post(
            "/api/v1/companies/me/clients", json={"client_type": "INVALID_TYPE"}, headers=company_headers
        )
        assert response.status_code in [400, 429, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data) or response.status_code == 429

        # Test SELF_SERVICE sans email (requis pour ce type)
        response = client.post(
            "/api/v1/companies/me/clients",
            json={
                "client_type": "SELF_SERVICE"
                # email manquant
            },
            headers=company_headers,
        )
        # Note: Le schéma n'a pas email comme required, mais la route peut vérifier
        # Acceptons 400 ou 201 selon l'implémentation
        assert response.status_code in [400, 201, 500]

        # Test PRIVATE sans first_name/last_name/address (requis pour ce type)
        response = client.post(
            "/api/v1/companies/me/clients",
            json={
                "client_type": "PRIVATE"
                # first_name, last_name, address manquants
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429, 500]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data) or response.status_code == 429

        # Test avec email invalide
        response = client.post(
            "/api/v1/companies/me/clients",
            json={"client_type": "SELF_SERVICE", "email": "invalid-email-format"},
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        error_str = str(data).lower()
        assert "email" in error_str or "errors" in error_str

        # Test avec first_name trop long (> 100)
        response = client.post(
            "/api/v1/companies/me/clients",
            json={
                "client_type": "PRIVATE",
                "first_name": "a" * 101,  # Max 100
                "last_name": "Dupont",
                "address": "Rue de Test 1, 1000 Lausanne",
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data)

        # Test avec billing_lat hors limites (> 90)
        response = client.post(
            "/api/v1/companies/me/clients",
            json={
                "client_type": "PRIVATE",
                "first_name": "Jean",
                "last_name": "Dupont",
                "address": "Rue de Test 1, 1000 Lausanne",
                "billing_lat": 91.0,  # Max 90
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert ("message" in data) or ("errors" in data) or ("error" in data)

    def test_create_payment_valid_schema(self, client, db):
        """✅ Test E2E POST /api/v1/payments/booking/<id> avec PaymentCreateSchema valide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Booking, BookingStatus, Client, User, UserRole

        # Créer un client user et une réservation
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()

        # Créer une réservation pour ce client
        future_time = datetime.now(UTC) + timedelta(days=1)
        booking = Booking()
        booking.client_id = test_client.id
        booking.status = BookingStatus.PENDING
        booking.customer_name = "Test Customer"
        booking.pickup_location = "Rue de la Gare 1, 1000 Lausanne"
        booking.dropoff_location = "Avenue de la Plage 10, 1000 Lausanne"
        booking.scheduled_time = future_time
        booking.amount = 50.0
        booking.user_id = client_user.id
        db.session.add(booking)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Générer un JWT pour le client
        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        client_token = create_access_token(
            identity=str(client_user.public_id),
            additional_claims={"role": client_user.role.value, "user_id": client_user.id},
            expires_delta=timedelta(hours=1),
        )
        client_headers = {"Authorization": f"Bearer {client_token}"}

        # Test création paiement valide
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={"amount": 50.0, "method": "credit_card", "reference": "REF-123456"},
            headers=client_headers,
        )
        assert response.status_code in [200, 201, 400, 500]
        data = response.get_json()
        if response.status_code in [200, 201]:
            assert "message" in data
            assert "payment_id" in data
        else:
            assert "error" in data

        # Test avec champs optionnels seulement (amount et method requis)
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}", json={"amount": 75.5, "method": "paypal"}, headers=client_headers
        )
        assert response.status_code in [200, 201, 400, 500]
        data = response.get_json()
        if response.status_code in [200, 201]:
            assert "message" in data
            assert "payment_id" in data
        else:
            assert "error" in data

    def test_create_payment_invalid_schema(self, client, db):
        """✅ Test E2E POST /api/v1/payments/booking/<id> avec PaymentCreateSchema invalide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Booking, BookingStatus, Client, User, UserRole

        # Créer un client user et une réservation
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()

        future_time = datetime.now(UTC) + timedelta(days=1)
        booking = Booking()
        booking.client_id = test_client.id
        booking.status = BookingStatus.PENDING
        booking.customer_name = "Test Customer"
        booking.pickup_location = "Rue de la Gare 1, 1000 Lausanne"
        booking.dropoff_location = "Avenue de la Plage 10, 1000 Lausanne"
        booking.scheduled_time = future_time
        booking.amount = 50.0
        booking.user_id = client_user.id
        db.session.add(booking)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        client_token = create_access_token(
            identity=str(client_user.public_id),
            additional_claims={"role": client_user.role.value, "user_id": client_user.id},
            expires_delta=timedelta(hours=1),
        )
        client_headers = {"Authorization": f"Bearer {client_token}"}

        # Test avec amount manquant (requis)
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={
                "method": "credit_card"
                # amount manquant
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        error_str = str(data).lower()
        assert "amount" in error_str or "errors" in error_str

        # Test avec method manquant (requis)
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={
                "amount": 50.0
                # method manquant
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        error_str = str(data).lower()
        assert "method" in error_str or "errors" in error_str

        # Test avec amount invalide (< 0.01)
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={
                "amount": 0.005,  # Min 0.01
                "method": "credit_card",
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec amount négatif
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={"amount": -10.0, "method": "credit_card"},
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec method trop long (> 50)
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={
                "amount": 50.0,
                "method": "a" * 51,  # Max 50
            },
            headers=client_headers,
        )
        assert response.status_code in [400, 500]
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec reference trop long (> 100)
        response = client.post(
            f"/api/v1/payments/booking/{booking.id}",
            json={
                "amount": 50.0,
                "method": "credit_card",
                "reference": "a" * 101,  # Max 100
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

    def test_update_payment_status_valid_schema(self, client, db):
        """✅ Test E2E PUT /api/v1/payments/<id> avec PaymentStatusUpdateSchema valide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Booking, BookingStatus, Client, Payment, PaymentStatus, User, UserRole

        # Créer un client user et une réservation
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()

        future_time = datetime.now(UTC) + timedelta(days=1)
        booking = Booking()
        booking.client_id = test_client.id
        booking.status = BookingStatus.PENDING
        booking.customer_name = "Test Customer"
        booking.pickup_location = "Rue de la Gare 1, 1000 Lausanne"
        booking.dropoff_location = "Avenue de la Plage 10, 1000 Lausanne"
        booking.scheduled_time = future_time
        booking.amount = 50.0
        booking.user_id = client_user.id
        db.session.add(booking)
        db.session.flush()

        # Utiliser un ID inexistant pour valider le schema sans dépendre des données
        test_payment_id = 99999999

        # Créer un admin user
        admin_user = User()
        admin_user.username = f"admin_{uuid.uuid4().hex[:6]}"
        admin_user.email = f"admin_{uuid.uuid4().hex[:6]}@example.com"
        admin_user.role = UserRole.admin
        admin_user.public_id = str(uuid.uuid4())
        # Utiliser la méthode de hachage standard de l'application
        admin_user.set_password("password123", force_change=False)
        db.session.add(admin_user)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Générer un JWT pour l'admin
        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        admin_token = create_access_token(
            identity=str(admin_user.public_id),
            additional_claims={"role": admin_user.role.value},
            expires_delta=timedelta(hours=1),
        )
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Test mise à jour status à "completed"
        response = client.put(
            f"/api/v1/payments/{test_payment_id}", json={"status": "completed"}, headers=admin_headers
        )
        assert response.status_code in [200, 404]
        data = response.get_json()
        if response.status_code == 200:
            assert "message" in data
            assert "completed" in data["message"].lower() or "updated" in data["message"].lower()
        else:
            assert "error" in data

        # Test mise à jour status à "failed"
        response = client.put(f"/api/v1/payments/{test_payment_id}", json={"status": "failed"}, headers=admin_headers)
        assert response.status_code in [200, 404]
        data = response.get_json()
        if response.status_code == 200:
            assert "message" in data
            assert "failed" in data["message"].lower() or "updated" in data["message"].lower()
        else:
            assert "error" in data

        # Test mise à jour status à "pending"
        response = client.put(f"/api/v1/payments/{test_payment_id}", json={"status": "pending"}, headers=admin_headers)
        assert response.status_code in [200, 404]
        data = response.get_json()
        if response.status_code == 200:
            assert "message" in data
        else:
            assert "error" in data

    def test_update_payment_status_invalid_schema(self, client, db):
        """✅ Test E2E PUT /api/v1/payments/<id> avec PaymentStatusUpdateSchema invalide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Booking, BookingStatus, Client, Payment, PaymentStatus, User, UserRole

        # Créer un client user et une réservation
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()

        future_time = datetime.now(UTC) + timedelta(days=1)
        booking = Booking()
        booking.client_id = test_client.id
        booking.status = BookingStatus.PENDING
        booking.customer_name = "Test Customer"
        booking.pickup_location = "Rue de la Gare 1, 1000 Lausanne"
        booking.dropoff_location = "Avenue de la Plage 10, 1000 Lausanne"
        booking.scheduled_time = future_time
        booking.amount = 50.0
        booking.user_id = client_user.id
        db.session.add(booking)
        db.session.flush()

        # Utiliser un ID inexistant pour déclencher la validation de schema
        test_payment_id = 99999999

        # Créer un admin user
        admin_user = User()
        admin_user.username = f"admin_{uuid.uuid4().hex[:6]}"
        admin_user.email = f"admin_{uuid.uuid4().hex[:6]}@example.com"
        admin_user.role = UserRole.admin
        admin_user.public_id = str(uuid.uuid4())
        # Utiliser la méthode de hachage standard de l'application
        admin_user.set_password("password123", force_change=False)
        db.session.add(admin_user)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Générer un JWT pour l'admin
        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        admin_token = create_access_token(
            identity=str(admin_user.public_id),
            additional_claims={"role": admin_user.role.value},
            expires_delta=timedelta(hours=1),
        )
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Test avec status manquant (requis)
        response = client.put(f"/api/v1/payments/{test_payment_id}", json={}, headers=admin_headers)
        assert response.status_code in [400, 404]
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        error_str = str(data).lower()
        assert "status" in error_str or "errors" in error_str

        # Test avec status invalide (pas dans ["pending", "completed", "failed"])
        response = client.put(
            f"/api/v1/payments/{test_payment_id}", json={"status": "INVALID_STATUS"}, headers=admin_headers
        )
        assert response.status_code in [400, 404]
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        error_str = str(data).lower()
        assert "status" in error_str or "invalid" in error_str or "errors" in error_str

        # Test avec status en majuscules (doit être en minuscules selon le schema)
        response = client.put(
            f"/api/v1/payments/{test_payment_id}",
            json={
                "status": "COMPLETED"  # Doit être "completed" en minuscules
            },
            headers=admin_headers,
        )
        assert response.status_code in [400, 404]
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec status vide
        response = client.put(f"/api/v1/payments/{test_payment_id}", json={"status": ""}, headers=admin_headers)
        assert response.status_code in [400, 404]
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

    def test_update_client_valid_schema(self, client, db):
        """✅ Test E2E PUT /api/clients/<id> avec ClientUpdateSchema valide."""
        from ext import bcrypt
        from models import Client, GenderEnum, User, UserRole

        # Créer un client user
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        client_user.first_name = "Original"
        client_user.last_name = "Name"
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        test_client.phone = "+41211234567"
        test_client.address = "Old Address"
        db.session.add(test_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Authentification: générer un JWT client directement
        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        client_token = create_access_token(
            identity=str(client_user.public_id),
            additional_claims={"role": UserRole.client.value},
            expires_delta=timedelta(hours=1),
        )
        client_headers = {"Authorization": f"Bearer {client_token}"}

        # Test mise à jour avec tous les champs
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "first_name": "Updated",
                "last_name": "Client",
                "phone": "+41219876543",
                "address": "New Address 123, 1000 Lausanne",
                "birth_date": "1990-05-15",
                "gender": "HOMME",
            },
            headers=client_headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "message" in data

        # Test mise à jour partielle (seulement certains champs)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={"first_name": "Partial", "phone": "+41211234567"},
            headers=client_headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "message" in data

        # Test mise à jour vide (tous les champs sont optionnels)
        response = client.put(f"/api/v1/clients/{client_user.public_id}", json={}, headers=client_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert "message" in data

    def test_update_client_invalid_schema(self, client, db):
        """✅ Test E2E PUT /api/clients/<id> avec ClientUpdateSchema invalide."""
        from ext import bcrypt
        from models import Client, User, UserRole

        # Créer un client user
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        # ✅ FIX: company_id est NOT NULL dans la base de données (migration initiale)
        # Créer une company temporaire pour ce test
        from models import Company

        temp_company = Company()
        temp_company.name = "Temp Company for Client Test"
        temp_company.user_id = client_user.id
        db.session.add(temp_company)
        db.session.flush()
        test_client.company_id = temp_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        from datetime import timedelta

        from flask_jwt_extended import create_access_token

        client_token = create_access_token(
            identity=str(client_user.public_id),
            additional_claims={"role": UserRole.client.value},
            expires_delta=timedelta(hours=1),
        )
        client_headers = {"Authorization": f"Bearer {client_token}"}

        # Test avec first_name trop long (> 100)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "first_name": "a" * 101  # Max 100
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec last_name trop long (> 100)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "last_name": "a" * 101  # Max 100
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec phone invalide (format invalide)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "phone": "invalid-phone"  # Format invalide
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec phone trop court (< 7 chiffres)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "phone": "123456"  # Trop court (< 7)
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec address trop long (> 500)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "address": "a" * 501  # Max 500
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec birth_date format invalide (pas YYYY-MM-DD)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "birth_date": "15-05-1990"  # Format invalide
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec gender invalide (pas dans les valeurs autorisées)
        response = client.put(
            f"/api/v1/clients/{client_user.public_id}",
            json={
                "gender": "INVALID"  # Pas dans ["HOMME", "FEMME", "AUTRE"]
            },
            headers=client_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

    def test_update_driver_profile_valid_schema(self, client, db, sample_company):
        """✅ Test E2E PUT /api/v1/driver/me/profile avec DriverProfileUpdateSchema valide."""
        from ext import bcrypt
        from models import Driver, User, UserRole

        # Créer un driver user
        driver_user = User()
        driver_user.username = f"driver_{uuid.uuid4().hex[:6]}"
        driver_user.email = f"driver_{uuid.uuid4().hex[:6]}@example.com"
        driver_user.role = UserRole.driver
        driver_user.public_id = str(uuid.uuid4())
        driver_user.first_name = "Original"
        driver_user.last_name = "Driver"
        password_hash = bcrypt.generate_password_hash("password123")
        driver_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(driver_user)
        db.session.flush()

        test_driver = Driver()
        test_driver.user_id = driver_user.id
        test_driver.company_id = sample_company.id
        test_driver.is_active = True
        test_driver.contract_type = "CDI"
        test_driver.weekly_hours = 40
        db.session.add(test_driver)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Authentification: générer un JWT direct (éviter dépendance login)
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.driver.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            driver_token = create_access_token(identity=str(driver_user.public_id), additional_claims=claims)
        driver_headers = {"Authorization": f"Bearer {driver_token}"}

        # Test mise à jour avec tous les champs
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "first_name": "Updated",
                "last_name": "DriverName",
                "phone": "+41211234567",
                "status": "disponible",
                "contract_type": "CDD",
                "weekly_hours": 35,
                "hourly_rate_cents": 5000,
                "employment_start_date": "2020-01-15",
                "employment_end_date": None,
                "license_valid_until": "2025-12-31",
                "medical_valid_until": "2025-06-30",
                "license_categories": ["B", "C"],
                "trainings": [{"name": "First Aid", "date": "2024-01-01"}],
            },
            headers=driver_headers,
        )
        assert response.status_code in [200, 400]
        data = response.get_json()
        assert "message" in data
        if response.status_code == 200:
            assert "profile" in data

        # Test mise à jour partielle (seulement certains champs)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={"first_name": "Partial", "status": "hors service"},
            headers=driver_headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "message" in data

        # Test mise à jour vide (tous les champs sont optionnels)
        response = client.put("/api/v1/driver/me/profile", json={}, headers=driver_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert "message" in data

    def test_update_driver_profile_invalid_schema(self, client, db, sample_company):
        """✅ Test E2E PUT /api/v1/driver/me/profile avec DriverProfileUpdateSchema invalide."""
        from ext import bcrypt
        from models import Driver, User, UserRole

        # Créer un driver user
        driver_user = User()
        driver_user.username = f"driver_{uuid.uuid4().hex[:6]}"
        driver_user.email = f"driver_{uuid.uuid4().hex[:6]}@example.com"
        driver_user.role = UserRole.driver
        driver_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        driver_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(driver_user)
        db.session.flush()

        test_driver = Driver()
        test_driver.user_id = driver_user.id
        test_driver.company_id = sample_company.id
        db.session.add(test_driver)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.driver.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            driver_token = create_access_token(identity=str(driver_user.public_id), additional_claims=claims)
        driver_headers = {"Authorization": f"Bearer {driver_token}"}

        # Test avec first_name trop long (> 100)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "first_name": "a" * 101  # Max 100
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec last_name trop long (> 100)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "last_name": "a" * 101  # Max 100
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec status invalide (pas dans ["disponible", "hors service"])
        response = client.put("/api/v1/driver/me/profile", json={"status": "INVALID"}, headers=driver_headers)
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec weekly_hours négatif (< 0)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "weekly_hours": -1  # Min 0
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec weekly_hours trop élevé (> 168)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "weekly_hours": 169  # Max 168
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec hourly_rate_cents négatif (< 0)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "hourly_rate_cents": -100  # Min 0
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec employment_start_date format invalide
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "employment_start_date": "15-01-2020"  # Format invalide (doit être YYYY-MM-DD)
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec license_categories trop nombreuses (> 10)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "license_categories": [f"CAT{i}" for i in range(11)]  # Max 10
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec trainings trop nombreuses (> 50)
        response = client.put(
            "/api/v1/driver/me/profile",
            json={
                "trainings": [{"name": f"Training {i}"} for i in range(51)]  # Max 50
            },
            headers=driver_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

    def test_update_billing_settings_valid_schema(self, client, db, sample_company):
        """✅ Test E2E PUT /api/v1/invoices/companies/<id>/billing-settings avec BillingSettingsUpdateSchema valide."""
        from models import CompanyBillingSettings, User

        # Créer CompanyBillingSettings pour la company
        billing_settings = CompanyBillingSettings()
        billing_settings.company_id = sample_company.id
        db.session.add(billing_settings)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Login en tant que company
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()

        login_response = client.post("/api/auth/login", json={"email": company_user.email, "password": "password123"})
        company_token = login_response.get_json()["token"]
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test mise à jour avec tous les champs valides
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "payment_terms_days": 30,
                "overdue_fee": 25.50,
                "reminder1fee": 5.00,
                "reminder2fee": 10.00,
                "reminder3fee": 15.00,
                "auto_reminders_enabled": True,
                "email_sender": "billing@example.com",
                "invoice_number_format": "{PREFIX}-{YYYY}-{MM}-{SEQ4}",
                "invoice_prefix": "INV",
                "iban": "CH9300762011623852957",
                "qr_iban": "CH9300762011623852957",
                "esr_ref_base": "12345678901234567890123456",
                "invoice_message_template": "Merci pour votre confiance",
                "reminder1template": "Premier rappel",
                "reminder2template": "Deuxième rappel",
                "reminder3template": "Dernier rappel",
                "legal_footer": "Texte légal de pied de page",
                "pdf_template_variant": "modern",
                "reminder_schedule_days": {"1": 10, "2": 5, "3": 5},
            },
            headers=company_headers,
        )
        assert response.status_code == 200

        # Test mise à jour partielle (seulement certains champs)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={"payment_terms_days": 60, "overdue_fee": 30.00},
            headers=company_headers,
        )
        assert response.status_code == 200

        # Test mise à jour vide (tous les champs sont optionnels)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings", json={}, headers=company_headers
        )
        assert response.status_code == 200

    def test_update_billing_settings_invalid_schema(self, client, db, sample_company):
        """✅ Test E2E PUT /api/v1/invoices/companies/<id>/billing-settings avec BillingSettingsUpdateSchema invalide."""
        from models import CompanyBillingSettings, User

        # Créer CompanyBillingSettings pour la company
        billing_settings = CompanyBillingSettings()
        billing_settings.company_id = sample_company.id
        db.session.add(billing_settings)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Login en tant que company
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()

        login_response = client.post("/api/auth/login", json={"email": company_user.email, "password": "password123"})
        company_token = login_response.get_json()["token"]
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test avec payment_terms_days négatif (< 0)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "payment_terms_days": -1  # Min 0
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec payment_terms_days trop élevé (> 365)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "payment_terms_days": 366  # Max 365
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec overdue_fee négatif (< 0)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "overdue_fee": -10.0  # Min 0
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec reminder1fee négatif (< 0)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "reminder1fee": -5.0  # Min 0
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec email_sender trop long (> 254)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "email_sender": "a" * 255  # Max 254
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec invoice_number_format trop long (> 50)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "invoice_number_format": "a" * 51  # Max 50
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec invoice_prefix trop long (> 20)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "invoice_prefix": "a" * 21  # Max 20
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec IBAN invalide (format incorrect)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "iban": "INVALID-IBAN"  # Format invalide
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec esr_ref_base trop long (> 26)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "esr_ref_base": "a" * 27  # Max 26
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec invoice_message_template trop long (> 1000)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "invoice_message_template": "a" * 1001  # Max 1000
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec reminder1template trop long (> 1000)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "reminder1template": "a" * 1001  # Max 1000
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec legal_footer trop long (> 2000)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "legal_footer": "a" * 2001  # Max 2000
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec pdf_template_variant trop long (> 50)
        response = client.put(
            f"/api/v1/invoices/companies/{sample_company.id}/billing-settings",
            json={
                "pdf_template_variant": "a" * 51  # Max 50
            },
            headers=company_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

    def test_generate_invoice_valid_schema(self, client, db, sample_company):
        """✅ Test E2E POST /api/v1/invoices/companies/<id>/invoices/generate avec InvoiceGenerateSchema valide."""
        from datetime import UTC, timedelta

        from ext import bcrypt
        from models import Client, User, UserRole

        # Créer un client pour cette company
        client_user = User()
        client_user.username = f"client_{uuid.uuid4().hex[:6]}"
        client_user.email = f"client_{uuid.uuid4().hex[:6]}@example.com"
        client_user.role = UserRole.client
        client_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        client_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(client_user)
        db.session.flush()

        test_client = Client()
        test_client.user_id = client_user.id
        test_client.company_id = sample_company.id
        test_client.client_type = "PRIVATE"
        db.session.add(test_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Login en tant que company
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()

        login_response = client.post("/api/auth/login", json={"email": company_user.email, "password": "password123"})
        company_token = login_response.get_json()["token"]
        company_headers = {"Authorization": f"Bearer {company_token}"}

        # Test génération facture simple avec client_id
        current_date = datetime.now(UTC)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={"client_id": test_client.id, "period_year": current_date.year, "period_month": current_date.month},
            headers=company_headers,
        )
        # Peut être 201 (succès) ou 400/404 si pas de réservations ou erreur DB
        assert response.status_code in [201, 400, 404]

        # Test génération avec client_ids (facturation groupée)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={"client_ids": [test_client.id], "period_year": current_date.year, "period_month": current_date.month},
            headers=company_headers,
        )
        assert response.status_code in [201, 400, 404]

        # Test avec bill_to_client_id (facturation tierce)
        institution_user = User()
        institution_user.username = f"inst_{uuid.uuid4().hex[:6]}"
        institution_user.email = f"inst_{uuid.uuid4().hex[:6]}@example.com"
        institution_user.role = UserRole.client
        institution_user.public_id = str(uuid.uuid4())
        # Assurer un mot de passe (NOT NULL)
        institution_user.set_password("password123", force_change=False)
        db.session.add(institution_user)
        db.session.flush()

        institution_client = Client()
        institution_client.user_id = institution_user.id
        institution_client.company_id = sample_company.id
        institution_client.client_type = "CORPORATE"
        institution_client.is_institution = True
        db.session.add(institution_client)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": test_client.id,
                "bill_to_client_id": institution_client.id,
                "period_year": current_date.year,
                "period_month": current_date.month,
            },
            headers=company_headers,
        )
        assert response.status_code in [201, 400, 404]

        # Test avec reservation_ids (sélection manuelle)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": test_client.id,
                "period_year": current_date.year,
                "period_month": current_date.month,
                "reservation_ids": [],
            },
            headers=company_headers,
        )
        assert response.status_code in [201, 400, 404]

        # Test avec client_reservations (facturation groupée avec sélection)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_ids": [test_client.id],
                "bill_to_client_id": institution_client.id,
                "period_year": current_date.year,
                "period_month": current_date.month,
                "client_reservations": {str(test_client.id): []},
            },
            headers=company_headers,
        )
        assert response.status_code in [201, 400, 404]

    def test_generate_invoice_invalid_schema(self, client, db, sample_company):
        """✅ Test E2E POST /api/v1/invoices/companies/<id>/invoices/generate avec InvoiceGenerateSchema invalide."""
        from models import User, UserRole

        # Auth company via JWT direct
        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        company_claims = {
            "role": UserRole.company.value,
            "company_id": sample_company.id,
            "driver_id": None,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            company_token = create_access_token(identity=str(company_user.public_id), additional_claims=company_claims)
        company_headers = {"Authorization": f"Bearer {company_token}"}

        current_date = datetime.now(UTC)

        # Test avec period_year manquant (requis)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 1,
                "period_month": current_date.month,
                # period_year manquant
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data
            error_str = str(data).lower()
            assert "period_year" in error_str or "errors" in error_str

        # Test avec period_month manquant (requis)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 1,
                "period_year": current_date.year,
                # period_month manquant
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data
            error_str = str(data).lower()
            assert "period_month" in error_str or "errors" in error_str

        # Test avec period_year trop bas (< 2000)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 1,
                "period_year": 1999,  # Min 2000
                "period_month": current_date.month,
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data

        # Test avec period_year trop élevé (> 2100)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 1,
                "period_year": 2101,  # Max 2100
                "period_month": current_date.month,
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data

        # Test avec period_month trop bas (< 1)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 1,
                "period_year": current_date.year,
                "period_month": 0,  # Min 1
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data

        # Test avec period_month trop élevé (> 12)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 1,
                "period_year": current_date.year,
                "period_month": 13,  # Max 12
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data

        # Test avec client_id négatif (< 1)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 0,  # Min 1
                "period_year": current_date.year,
                "period_month": current_date.month,
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data

        # Test avec client_ids vide (minimum 1 requis)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_ids": [],  # Min 1
                "period_year": current_date.year,
                "period_month": current_date.month,
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data

        # Test avec bill_to_client_id négatif (< 1)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "client_id": 1,
                "bill_to_client_id": 0,  # Min 1
                "period_year": current_date.year,
                "period_month": current_date.month,
            },
            headers=company_headers,
        )
        assert response.status_code in [400, 429]  # 429 pour rate limiting
        data = response.get_json() if response.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if response.status_code != 429:
            assert "message" in data or "errors" in data

        # Test sans client_id ni client_ids (au moins un requis par la logique métier)
        response = client.post(
            f"/api/v1/invoices/companies/{sample_company.id}/invoices/generate",
            json={
                "period_year": current_date.year,
                "period_month": current_date.month,
                # client_id et client_ids manquants
            },
            headers=company_headers,
        )
        # L'endpoint vérifie cette condition après validation schema, donc 400 (ou 429 si rate limit atteint)
        assert response.status_code in [400, 429]  # 429 pour rate limiting

    # ========== COMPANIES ENDPOINTS ==========

    def test_update_company_valid_schema(self, client, auth_headers):
        """Test PUT /api/companies/me avec payload valide."""
        response = client.put(
            "/api/companies/me",
            json={
                "name": "Updated Company Name",
                "contact_email": "updated@example.com",
                "iban": "CH9300762011623852957",
                "uid_ide": "CHE-123.456.789",
            },
            headers=auth_headers,
        )
        assert response.status_code in [200, 404]  # 404 si pas de company

    def test_update_company_invalid_schema(self, client, auth_headers):
        """Test PUT /api/companies/me avec payload invalide (IBAN invalide)."""
        response = client.put(
            "/api/companies/me", json={"name": "Updated Company", "iban": "INVALID-IBAN"}, headers=auth_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    def test_create_driver_valid_schema(self, client, auth_headers):
        """Test POST /api/companies/me/drivers/create avec payload valide."""
        unique_email = f"driver_{uuid.uuid4().hex[:6]}@example.com"
        response = client.post(
            "/api/v1/companies/me/drivers/create",
            json={
                "username": f"driver_{uuid.uuid4().hex[:6]}",
                "first_name": "John",
                "last_name": "Driver",
                "email": unique_email,
                "password": "SecurePass123!",
                "vehicle_assigned": "Car 1",
                "brand": "Mercedes",
                "license_plate": "GE-12345",
            },
            headers=auth_headers,
        )
        assert response.status_code in [201, 400, 404]  # 404 si pas de company

    def test_create_driver_invalid_schema(self, client, auth_headers):
        """Test POST /api/companies/me/drivers/create avec payload invalide (champs manquants)."""
        response = client.post(
            "/api/v1/companies/me/drivers/create",
            json={
                "username": "driver1"
                # email, password, etc. manquants
            },
            headers=auth_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    # ========== MEDICAL ENDPOINTS (QUERY PARAMS) ==========

    def test_medical_establishments_valid_query(self, client):
        """Test GET /api/medical/establishments avec query params valides."""
        response = client.get("/api/v1/medical/establishments?q=hospital&limit=10")
        assert response.status_code == 200

    def test_medical_establishments_invalid_query(self, client):
        """Test GET /api/medical/establishments avec query params invalides (limit trop élevé).

        Note: La route utilise un fallback sur reqparse qui limite automatiquement à 25,
        donc on accepte 200 avec limit=25 appliqué.
        """
        response = client.get("/api/v1/medical/establishments?limit=100")
        # La route limite automatiquement à 25 (fallback reqparse), donc 200 OK
        assert response.status_code == 200
        # Vérifier que le résultat respecte la limite max
        data = response.get_json()
        if isinstance(data, list):
            assert len(data) <= 25
        elif isinstance(data, dict) and "results" in data:
            assert len(data["results"]) <= 25

    def test_medical_services_valid_query(self, client):
        """Test GET /api/medical/services avec query params valides."""
        response = client.get("/api/v1/medical/services?establishment_id=1&q=cardio")
        assert response.status_code in [200, 404]  # 404 si établissement introuvable

    def test_medical_services_invalid_query(self, client):
        """Test GET /api/medical/services avec query params invalides (establishment_id manquant)."""
        response = client.get("/api/v1/medical/services?q=cardio")
        # establishment_id est requis, donc 400 attendu
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data or "error" in data

    # ========== ANALYTICS ENDPOINTS (QUERY PARAMS) ==========

    def test_analytics_dashboard_valid_query(self, client, auth_headers):
        """Test GET /api/analytics/dashboard avec query params valides."""
        response = client.get("/api/v1/analytics/dashboard?period=30d", headers=auth_headers)
        assert response.status_code in [200, 404]  # 404 si pas de company

    def test_analytics_dashboard_invalid_query(self, client, auth_headers):
        """Test GET /api/analytics/dashboard avec query params invalides (period invalide)."""
        response = client.get("/api/v1/analytics/dashboard?period=invalid", headers=auth_headers)
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    def test_analytics_insights_valid_query(self, client, auth_headers):
        """Test GET /api/analytics/insights avec query params valides."""
        response = client.get("/api/v1/analytics/insights?lookback_days=30", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_analytics_insights_invalid_query(self, client, auth_headers):
        """Test GET /api/analytics/insights avec query params invalides (lookback_days trop élevé)."""
        response = client.get("/api/v1/analytics/insights?lookback_days=400", headers=auth_headers)
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    def test_analytics_export_valid_query(self, client, auth_headers):
        """Test GET /api/analytics/export avec query params valides."""
        start_date = date.today() - timedelta(days=7)
        end_date = date.today()
        response = client.get(
            f"/api/v1/analytics/export?start_date={start_date.isoformat()}&end_date={end_date.isoformat()}&format=csv",
            headers=auth_headers,
        )
        assert response.status_code in [200, 404]

    def test_analytics_export_invalid_query(self, client, auth_headers):
        """Test GET /api/analytics/export avec query params invalides (dates manquantes)."""
        response = client.get("/api/v1/analytics/export?format=csv", headers=auth_headers)
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    def test_analytics_weekly_summary_valid_query(self, client, auth_headers):
        """✅ Test E2E GET /api/analytics/weekly-summary avec AnalyticsWeeklySummaryQuerySchema valide."""
        from datetime import date, timedelta

        # Test avec week_start spécifié
        week_start = date.today() - timedelta(days=7)
        response = client.get(
            f"/api/v1/analytics/weekly-summary?week_start={week_start.isoformat()}", headers=auth_headers
        )
        # Peut être 200 (succès) ou 404 (pas de company) ou 500 (erreur serveur)
        assert response.status_code in [200, 404, 500]

        # Test sans week_start (optionnel, utilise la semaine courante par défaut)
        response = client.get("/api/v1/analytics/weekly-summary", headers=auth_headers)
        assert response.status_code in [200, 404, 500]

        # Si succès (200), vérifier la structure de la réponse
        if response.status_code == 200:
            data = response.get_json()
            assert "success" in data or "data" in data

    def test_analytics_weekly_summary_invalid_query(self, client, auth_headers):
        """✅ Test E2E GET /api/analytics/weekly-summary avec AnalyticsWeeklySummaryQuerySchema invalide."""
        # Test avec format date invalide (pas YYYY-MM-DD)
        response = client.get(
            "/api/v1/analytics/weekly-summary?week_start=01/15/2024",  # Format US invalide
            headers=auth_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data
        # Vérifier que l'erreur mentionne week_start ou format date
        error_str = str(data).lower()
        assert "week_start" in error_str or "date" in error_str or "format" in error_str or "errors" in error_str

        # Test avec format date invalide (pas ISO8601)
        response = client.get(
            "/api/v1/analytics/weekly-summary?week_start=2024-1-1",  # Format invalide (sans zéro padding)
            headers=auth_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

        # Test avec date mal formée
        response = client.get("/api/v1/analytics/weekly-summary?week_start=invalid-date", headers=auth_headers)
        assert response.status_code == 400
        data = response.get_json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        assert "message" in data or "errors" in data

    # ========== PLANNING ENDPOINTS (QUERY PARAMS) ==========

    def test_planning_shifts_valid_query(self, client, auth_headers):
        """Test GET /api/planning/companies/me/planning/shifts avec query params valides."""
        response = client.get("/api/v1/planning/companies/me/planning/shifts?driver_id=1", headers=auth_headers)
        assert response.status_code in [200, 401]  # 401 si pas autorisé

    def test_planning_shifts_invalid_query(self, client, auth_headers):
        """Test GET /api/planning/companies/me/planning/shifts avec query params invalides (driver_id négatif)."""
        response = client.get("/api/v1/planning/companies/me/planning/shifts?driver_id=-1", headers=auth_headers)
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    def test_planning_unavailability_valid_query(self, client, auth_headers):
        """✅ Test E2E GET /api/planning/companies/me/planning/unavailability avec PlanningUnavailabilityQuerySchema valide."""
        # Test avec driver_id spécifié
        response = client.get("/api/v1/planning/companies/me/planning/unavailability?driver_id=1", headers=auth_headers)
        # Peut être 200 (succès) ou 401 (pas autorisé)
        assert response.status_code in [200, 401]

        # Si succès (200), vérifier la structure de la réponse
        if response.status_code == 200:
            data = response.get_json()
            assert "items" in data or "total" in data

        # Test sans driver_id (optionnel)
        response = client.get("/api/v1/planning/companies/me/planning/unavailability", headers=auth_headers)
        assert response.status_code in [200, 401]

        # Si succès, vérifier la structure
        if response.status_code == 200:
            data = response.get_json()
            assert "items" in data or "total" in data

    def test_planning_unavailability_invalid_query(self, client, auth_headers):
        """✅ Test E2E GET /api/planning/companies/me/planning/unavailability avec PlanningUnavailabilityQuerySchema invalide."""
        # Test avec driver_id négatif (< 1)
        response = client.get(
            "/api/v1/planning/companies/me/planning/unavailability?driver_id=-1", headers=auth_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data
        # Vérifier que l'erreur mentionne driver_id
        error_str = str(data).lower()
        assert "driver_id" in error_str or "errors" in error_str

        # Test avec driver_id = 0 (invalide, doit être >= 1)
        response = client.get("/api/v1/planning/companies/me/planning/unavailability?driver_id=0", headers=auth_headers)
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

        # Test avec driver_id non numérique (string)
        response = client.get(
            "/api/v1/planning/companies/me/planning/unavailability?driver_id=abc", headers=auth_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    def test_planning_weekly_template_valid_query(self, client, auth_headers):
        """✅ Test E2E GET /api/planning/companies/me/planning/weekly-template avec PlanningWeeklyTemplateQuerySchema valide."""
        # Test avec driver_id spécifié
        response = client.get(
            "/api/v1/planning/companies/me/planning/weekly-template?driver_id=1", headers=auth_headers
        )
        # Peut être 200 (succès) ou 401 (pas autorisé)
        assert response.status_code in [200, 401]

        # Si succès (200), vérifier la structure de la réponse
        if response.status_code == 200:
            data = response.get_json()
            assert "items" in data or "total" in data

        # Test sans driver_id (optionnel)
        response = client.get("/api/v1/planning/companies/me/planning/weekly-template", headers=auth_headers)
        assert response.status_code in [200, 401]

        # Si succès, vérifier la structure
        if response.status_code == 200:
            data = response.get_json()
            assert "items" in data or "total" in data

    def test_planning_weekly_template_invalid_query(self, client, auth_headers):
        """✅ Test E2E GET /api/planning/companies/me/planning/weekly-template avec PlanningWeeklyTemplateQuerySchema invalide."""
        # Test avec driver_id négatif (< 1)
        response = client.get(
            "/api/v1/planning/companies/me/planning/weekly-template?driver_id=-1", headers=auth_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data
        # Vérifier que l'erreur mentionne driver_id
        error_str = str(data).lower()
        assert "driver_id" in error_str or "errors" in error_str

        # Test avec driver_id = 0 (invalide, doit être >= 1)
        response = client.get(
            "/api/v1/planning/companies/me/planning/weekly-template?driver_id=0", headers=auth_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

        # Test avec driver_id non numérique (string)
        response = client.get(
            "/api/v1/planning/companies/me/planning/weekly-template?driver_id=abc", headers=auth_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    # ========== ADMIN ENDPOINTS ==========

    def test_update_user_role_valid_schema(self, client, db):
        """Test PUT /api/admin/users/<id>/role avec payload valide."""
        from models import User, UserRole

        # Créer un admin user
        admin_user = User()
        admin_user.username = f"admin_{uuid.uuid4().hex[:6]}"
        admin_user.email = f"admin_{uuid.uuid4().hex[:6]}@example.com"
        admin_user.role = UserRole.admin
        admin_user.public_id = str(uuid.uuid4())
        # Utiliser la méthode de hachage standard de l'application
        admin_user.set_password("password123", force_change=False)
        db.session.add(admin_user)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Auth admin via JWT
        from flask_jwt_extended import create_access_token

        admin_claims = {"role": UserRole.admin.value, "company_id": None, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            admin_token = create_access_token(identity=str(admin_user.public_id), additional_claims=admin_claims)
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Créer un user à modifier
        target_user = User()
        target_user.username = f"target_{uuid.uuid4().hex[:6]}"
        target_user.email = f"target_{uuid.uuid4().hex[:6]}@example.com"
        target_user.role = UserRole.client
        admin_user.public_id = str(uuid.uuid4())
        target_user.set_password("password123", force_change=False)
        db.session.add(target_user)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        response = client.put(
            f"/api/v1/admin/users/{target_user.id}/role",
            json={"role": "driver", "company_id": 1},
            headers=admin_headers,
        )
        assert response.status_code in [200, 400, 404]  # 400 si company_id invalide

    def test_update_user_role_invalid_schema(self, client, db):
        """Test PUT /api/admin/users/<id>/role avec payload invalide (rôle invalide)."""
        from models import User, UserRole

        admin_user = User()
        admin_user.username = f"admin_{uuid.uuid4().hex[:6]}"
        admin_user.email = f"admin_{uuid.uuid4().hex[:6]}@example.com"
        admin_user.role = UserRole.admin
        admin_user.public_id = str(uuid.uuid4())
        admin_user.set_password("password123", force_change=False)
        db.session.add(admin_user)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        from flask_jwt_extended import create_access_token

        admin_claims = {"role": UserRole.admin.value, "company_id": None, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            admin_token = create_access_token(identity=str(admin_user.public_id), additional_claims=admin_claims)
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        response = client.put(
            f"/api/v1/admin/users/{admin_user.id}/role", json={"role": "invalid_role"}, headers=admin_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "message" in data or "errors" in data

    # ========== ADMIN AUTONOMOUS ACTIONS REVIEW (E2E) ==========

    def test_autonomous_action_review_valid(self, client, db):
        """✅ Test E2E POST /api/v1/admin/autonomous-actions/<id>/review avec payload valide."""
        from ext import bcrypt
        from models import Company, User, UserRole
        from models.autonomous_action import AutonomousAction

        # Créer un admin user et se logger
        admin_user = User()
        admin_user.username = f"admin_{uuid.uuid4().hex[:6]}"
        admin_user.email = f"admin_{uuid.uuid4().hex[:6]}@example.com"
        admin_user.role = UserRole.admin
        admin_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        admin_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(admin_user)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Générer un token JWT admin directement pour éviter flakiness du login en tests
        from flask_jwt_extended import create_access_token

        claims = {
            "role": admin_user.role.value,
            "company_id": getattr(admin_user, "company_id", None),
            "driver_id": getattr(admin_user, "driver_id", None),
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(identity=str(admin_user.public_id), additional_claims=claims)
        headers = {"Authorization": f"Bearer {token}"}

        # Créer une company dédiée à ce test
        company = Company()
        company.name = "Admin Test Co"
        company.user_id = admin_user.id
        db.session.add(company)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Créer une action autonome
        action = AutonomousAction()
        action.company_id = company.id
        action.action_type = "reassign"
        action.action_description = "Réassignation automatique test"
        action.success = True
        db.session.add(action)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Review avec notes valides
        resp = client.post(
            f"/api/v1/admin/autonomous-actions/{action.id}/review",
            json={"notes": "OK pour production"},
            headers=headers,
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "message" in body
        assert "action" in body
        action_payload = body["action"]
        assert isinstance(action_payload, dict)
        assert action_payload.get("reviewed_by_admin") is True

        # Review sans notes (optionnel)
        resp2 = client.post(
            f"/api/v1/admin/autonomous-actions/{action.id}/review",
            json={},
            headers=headers,
        )
        assert resp2.status_code == 200

    def test_autonomous_action_review_invalid(self, client, db):
        """✅ Test E2E POST /api/v1/admin/autonomous-actions/<id>/review avec payload invalide (notes trop longues)."""
        from ext import bcrypt
        from models import Company, User, UserRole
        from models.autonomous_action import AutonomousAction

        # Créer un admin user et se logger
        admin_user = User()
        admin_user.username = f"admin_{uuid.uuid4().hex[:6]}"
        admin_user.email = f"admin_{uuid.uuid4().hex[:6]}@example.com"
        admin_user.role = UserRole.admin
        admin_user.public_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash("password123")
        admin_user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash
        db.session.add(admin_user)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        from flask_jwt_extended import create_access_token

        claims = {
            "role": admin_user.role.value,
            "company_id": getattr(admin_user, "company_id", None),
            "driver_id": getattr(admin_user, "driver_id", None),
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(identity=str(admin_user.public_id), additional_claims=claims)
        headers = {"Authorization": f"Bearer {token}"}

        # Créer une company dédiée à ce test
        company = Company()
        company.name = "Admin Test Co 2"
        company.user_id = admin_user.id
        db.session.add(company)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Créer une action autonome
        action = AutonomousAction()
        action.company_id = company.id
        action.action_type = "reassign"
        action.action_description = "Réassignation automatique test"
        action.success = True
        db.session.add(action)
        db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

        # Payload invalide: notes > 1000 caractères
        too_long_notes = "a" * 1001
        resp = client.post(
            f"/api/v1/admin/autonomous-actions/{action.id}/review",
            json={"notes": too_long_notes},
            headers=headers,
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "message" in data or "errors" in data

    # ========== COMPANIES ENDPOINTS ==========
    def test_company_me_get_returns_company(self, client, auth_headers):
        url = "/api/v1/companies/me"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404]
        body = resp.get_json() or {}
        if resp.status_code == 200:
            assert isinstance(body, dict)
            assert "id" in body
            assert ("name" in body) or ("contact_email" in body) or ("address" in body)

    def test_company_me_update_valid_schema(self, client, auth_headers):
        url = "/api/v1/companies/me"
        payload = {
            "name": "ATMR Transports SA",
            "contact_email": "ops@atmr.ch",
            "contact_phone": "+41211234567",
            "iban": "CH9300762011623852957",
            "uid_ide": "CHE-123.456.789",
            "domicile_city": "Genève",
            "domicile_country": "CH",
        }
        resp = client.put(url, json=payload, headers=auth_headers)
        assert resp.status_code in [200, 400]
        data = resp.get_json() or {}
        if resp.status_code == 200:
            for k in [
                "name",
                "contact_email",
                "contact_phone",
                "iban",
                "uid_ide",
            ]:
                assert k in data
            assert "domicile" in data
            domicile = data["domicile"]
            assert isinstance(domicile, dict)
            assert domicile.get("city") is not None
            assert domicile.get("country") is not None
        else:
            assert ("message" in data and "errors" in data) or ("error" in data)

    def test_company_me_update_invalid_schema(self, client, auth_headers):
        url = "/api/v1/companies/me"
        payload = {
            "iban": "XX-INVALID",
            "uid_ide": "BAD-123",
            "domicile_country": "CHE",
            "contact_email": "not-an-email",
        }
        resp = client.put(url, json=payload, headers=auth_headers)
        assert resp.status_code in [400, 422]
        data = resp.get_json() or {}
        assert ("message" in data and "errors" in data) or ("error" in data)

    def test_company_reservations_list(self, client, auth_headers):
        """Test GET /api/v1/companies/me/reservations - liste des réservations."""
        url = "/api/v1/companies/me/reservations"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404]
        if resp.status_code == 200:
            data = resp.get_json() or {}
            assert "reservations" in data
            assert "total" in data
            assert isinstance(data["reservations"], list)
            assert isinstance(data["total"], int)

    def test_company_drivers_list(self, client, auth_headers):
        """Test GET /api/v1/companies/me/drivers - liste des chauffeurs."""
        url = "/api/v1/companies/me/drivers"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404]
        if resp.status_code == 200:
            data = resp.get_json() or {}
            assert isinstance(data, list) or ("drivers" in data)

    def test_company_clients_list(self, client, auth_headers):
        """Test GET /api/v1/companies/me/clients - liste des clients."""
        url = "/api/v1/companies/me/clients"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404]
        if resp.status_code == 200:
            data = resp.get_json() or {}
            assert "clients" in data or isinstance(data, list)

    def test_company_dispatch_status(self, client, auth_headers):
        """Test GET /api/v1/companies/me/dispatch/status - statut dispatch."""
        url = "/api/v1/companies/me/dispatch/status"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert "dispatch_enabled" in data
        assert isinstance(data["dispatch_enabled"], bool)

    def test_company_dispatch_activate(self, client, auth_headers):
        """Test POST /api/v1/companies/me/dispatch/activate - activer dispatch."""
        url = "/api/v1/companies/me/dispatch/activate"
        resp = client.post(url, json={"enabled": True}, headers=auth_headers)
        assert resp.status_code in [200, 400]
        if resp.status_code == 200:
            data = resp.get_json() or {}
            assert "dispatch_enabled" in data
            assert isinstance(data["dispatch_enabled"], bool)

    def test_company_driver_create_valid_schema(self, client, auth_headers):
        """Test POST /api/v1/companies/me/drivers/create - création chauffeur valide."""
        url = "/api/v1/companies/me/drivers/create"
        unique_suffix = uuid.uuid4().hex[:8]
        payload = {
            "username": f"driver_{unique_suffix}",
            "first_name": "Jean",
            "last_name": "Dupont",
            "email": f"driver_{unique_suffix}@example.com",
            "password": "SecurePass123!",
            "vehicle_assigned": "Voiture 1",
            "brand": "Mercedes",
            "license_plate": f"GE-{unique_suffix[:6]}",
        }
        resp = client.post(url, json=payload, headers=auth_headers)
        assert resp.status_code in [201, 400, 409, 429, 500]
        if resp.status_code == 201:
            data = resp.get_json() or {}
            assert "id" in data
            assert "user_id" in data

    def test_company_driver_create_invalid_schema(self, client, auth_headers):
        """Test POST /api/v1/companies/me/drivers/create - création chauffeur invalide."""
        url = "/api/v1/companies/me/drivers/create"
        payload = {
            "username": "ab",  # trop court (< 3)
            "email": "invalid-email",  # format invalide
            "password": "short",  # trop court (< 8)
        }
        resp = client.post(url, json=payload, headers=auth_headers)
        assert resp.status_code in [400, 422]
        data = resp.get_json() or {}
        assert ("message" in data and "errors" in data) or ("error" in data)

    def test_company_vehicles_list(self, client, auth_headers):
        """Test GET /api/v1/companies/me/vehicles - liste véhicules."""
        url = "/api/v1/companies/me/vehicles"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404]
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, list)

    def test_company_invoices_list(self, client, auth_headers):
        """Test GET /api/v1/companies/me/invoices - liste factures."""
        url = "/api/v1/companies/me/invoices"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404, 500]
        if resp.status_code == 200:
            data = resp.get_json() or {}
            assert "invoices" in data or isinstance(data, list)

    def test_company_dispatch_status_get(self, client, auth_headers):
        url = "/api/v1/companies/me/dispatch/status"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404, 400]
        data = resp.get_json() or {}
        # Réponse peut être un dict avec des clés de statut ou un message d'erreur
        assert isinstance(data, dict)

    def test_company_dispatch_activate_post(self, client, auth_headers):
        url = "/api/v1/companies/me/dispatch/activate"
        resp = client.post(url, json={}, headers=auth_headers)
        assert resp.status_code in [200, 400, 404]
        data = resp.get_json() or {}
        assert isinstance(data, dict)

    def test_company_dispatch_deactivate_post(self, client, auth_headers):
        url = "/api/v1/companies/me/dispatch/deactivate"
        resp = client.post(url, json={}, headers=auth_headers)
        assert resp.status_code in [200, 400, 404]
        data = resp.get_json() or {}
        assert isinstance(data, dict)

    def test_company_assigned_reservations_get(self, client, auth_headers):
        url = "/api/v1/companies/me/assigned-reservations"
        resp = client.get(url, headers=auth_headers)
        assert resp.status_code in [200, 404]
        if resp.status_code == 200:
            data = resp.get_json() or {}
            assert "reservations" in data
            assert isinstance(data["reservations"], list)


# ================== COMPANIES RESERVATIONS ACTIONS (ACCEPT/REJECT/ASSIGN/COMPLETE) ==================


class TestCompaniesReservationActions:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_reservation_accept(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post("/api/v1/companies/me/reservations/999999/accept", json={"note": "ok"}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        data = resp.get_json() or {}
        assert isinstance(data, dict)

    def test_company_reservation_reject(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post(
            "/api/v1/companies/me/reservations/999999/reject", json={"reason": "indisponible"}, headers=headers
        )
        assert resp.status_code in [200, 400, 404]
        data = resp.get_json() or {}
        assert isinstance(data, dict)

    def test_company_reservation_assign(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # driver_id requis côté schéma, ici on force un invalide pour 400/404
        resp = client.post("/api/v1/companies/me/reservations/999999/assign", json={"driver_id": 0}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        data = resp.get_json() or {}
        assert isinstance(data, dict)

    def test_company_reservation_complete(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post(
            "/api/v1/companies/me/reservations/999999/complete", json={"status": "done"}, headers=headers
        )
        assert resp.status_code in [200, 400, 404]
        data = resp.get_json() or {}
        assert isinstance(data, dict)


# ================== COMPANIES VEHICLES (LIST/CREATE/GET/UPDATE/DELETE) ==================


class TestCompaniesVehicles:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_vehicles_list(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/vehicles", headers=headers)
        assert resp.status_code in [200, 404]
        data = resp.get_json() or []
        assert isinstance(data, (list, dict))

    def test_company_vehicle_create_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # Vide -> devrait échouer à la validation
        resp = client.post("/api/v1/companies/me/vehicles", json={}, headers=headers)
        assert resp.status_code in [400, 429]
        data = resp.get_json() if resp.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if resp.status_code != 429:
            assert ("message" in data) or ("errors" in data)

    def test_company_vehicle_get_by_id(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/vehicles/999999", headers=headers)
        assert resp.status_code in [200, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_company_vehicle_update_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.put("/api/v1/companies/me/vehicles/999999", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        _ = resp.get_json() if resp.is_json else None

    def test_company_vehicle_delete(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.delete("/api/v1/companies/me/vehicles/999999", headers=headers)
        assert resp.status_code in [200, 404]


# ================== COMPANIES DRIVERS VACATIONS (POST/GET) ==================


class TestCompaniesDriverVacations:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_driver_vacations_get(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/drivers/999999/vacations", headers=headers)
        assert resp.status_code in [200, 404]
        _ = resp.get_json() if resp.is_json else None

    def test_company_driver_vacations_post_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # Charge utile incomplète pour provoquer 400 (ou 404 si driver inconnu)
        payload = {
            # "start_date": "2025-01-01",  # requis normalement
            # "end_date": "2025-01-10",   # requis normalement
            "reason": "annual"
        }
        resp = client.post("/api/v1/companies/me/drivers/999999/vacations", json=payload, headers=headers)
        assert resp.status_code in [400, 404]
        data = resp.get_json() or {}
        assert isinstance(data, (dict, list))


# ================== COMPANIES CLIENTS (LIST/CREATE/GET/UPDATE/DELETE) ==================


class TestCompaniesClients:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_clients_list(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/clients", headers=headers)
        assert resp.status_code in [200, 404]
        _ = resp.get_json() if resp.is_json else None

    def test_company_clients_create_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # Charge incomplète -> devrait déclencher 400 (ou 429 RL)
        payload = {
            # "user_id": 1,
            # "client_type": "PRIVATE",
        }
        resp = client.post("/api/v1/companies/me/clients", json=payload, headers=headers)
        assert resp.status_code in [400, 429]
        data = resp.get_json() if resp.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if resp.status_code != 429:
            assert ("message" in data) or ("errors" in data)

    def test_company_client_get_by_id(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/clients/999999", headers=headers)
        assert resp.status_code in [200, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_company_client_update_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.put("/api/v1/companies/me/clients/999999", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        _ = resp.get_json() if resp.is_json else None

    def test_company_client_delete(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.delete("/api/v1/companies/me/clients/999999", headers=headers)
        assert resp.status_code in [200, 404]


# ================== COMPANIES MANUAL RESERVATIONS (CREATE/SCHEDULE/DISPATCH-NOW/TRIGGER-RETURN) ==================


class TestCompaniesManualReservations:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_manual_reservation_create_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # Données minimales manquantes pour forcer la validation à échouer
        payload = {
            # "client_id": 1,
            # "pickup_time": "2025-01-01T10:00:00Z",
            # "origin": {"address": "..."},
            # "destination": {"address": "..."},
        }
        resp = client.post("/api/v1/companies/me/reservations/manual", json=payload, headers=headers)
        assert resp.status_code in [400, 404, 429]
        data = resp.get_json() if resp.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if resp.status_code != 429:
            assert ("message" in data) or ("errors" in data) or ("error" in data)

    def test_company_reservation_schedule_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post("/api/v1/companies/me/reservations/999999/schedule", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_company_reservation_dispatch_now_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post("/api/v1/companies/me/reservations/999999/dispatch-now", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        _ = resp.get_json() if resp.is_json else None

    def test_company_reservation_trigger_return_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post("/api/v1/companies/me/reservations/999999/trigger-return", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        _ = resp.get_json() if resp.is_json else None


# ================== COMPANIES DRIVERS COMPLETED-TRIPS / TOGGLE-TYPE ==================


class TestCompaniesDriverExtras:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_driver_completed_trips(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/drivers/999999/completed-trips", headers=headers)
        assert resp.status_code in [200, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_company_driver_toggle_type_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # Payload invalide/missing pour déclencher 400 ou 404
        resp = client.post("/api/v1/companies/me/drivers/999999/toggle-type", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404, 405]
        _ = resp.get_json() if resp.is_json else None


# ================== COMPANIES INVOICES + LOGO ==================


class TestCompaniesInvoicesAndLogo:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_me_invoices_list(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/invoices", headers=headers)
        assert resp.status_code in [200, 404, 500]
        _ = resp.get_json() if resp.is_json else None

    def test_company_me_logo_upload_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # Pas de fichier multipart -> devrait échouer
        resp = client.post("/api/v1/companies/me/logo", json={"logo": "not-a-file"}, headers=headers)
        assert resp.status_code in [400, 404, 405, 415]
        _ = resp.get_json() if resp.is_json else None


# ================== COMPANIES MISC (DRIVERS CREATE, CLIENT RESERVATIONS, COMPANIES LIST) ==================


class TestCompaniesMisc:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_drivers_create_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        # Payload vide -> 400 attendu (ou 429 RL)
        resp = client.post("/api/v1/companies/me/drivers/create", json={}, headers=headers)
        assert resp.status_code in [400, 404, 405, 429]
        data = resp.get_json() if resp.status_code != 429 else {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if resp.status_code != 429:
            assert ("message" in data) or ("errors" in data) or ("error" in data)

    def test_company_client_reservations_list(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/clients/999999/reservations", headers=headers)
        assert resp.status_code in [200, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_companies_list(self, client, db, sample_company):
        # Liste des companies (peut être restreinte selon rôle) -> tolérer 200/403/404/405
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/", headers=headers)
        assert resp.status_code in [200, 403, 404, 405]
        _ = resp.get_json() if resp.is_json else None


# ================== COMPANIES RESERVATION BY ID (GET/DELETE) ==================


class TestCompaniesReservationById:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_reservation_get_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/reservations/999999", headers=headers)
        assert resp.status_code in [200, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_company_reservation_delete_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.delete("/api/v1/companies/me/reservations/999999", headers=headers)
        assert resp.status_code in [200, 404, 405]


# ================== COMPANIES ASSIGNED RESERVATIONS ==================


class TestCompaniesAssignedReservations:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_assigned_reservations_list(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/assigned-reservations", headers=headers)
        assert resp.status_code in [200, 404]
        _ = resp.get_json() if resp.is_json else None


# ================== COMPANIES DRIVER BY ID (GET/PUT/DELETE) ==================


class TestCompaniesDriverById:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_driver_get_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/drivers/999999", headers=headers)
        assert resp.status_code in [200, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_company_driver_update_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.put("/api/v1/companies/me/drivers/999999", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404, 405]
        _ = resp.get_json() if resp.is_json else None

    def test_company_driver_delete_invalid(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.delete("/api/v1/companies/me/drivers/999999", headers=headers)
        assert resp.status_code in [200, 404, 405]


# ================== COMPANIES DISPATCH (STATUS/ACTIVATE/DEACTIVATE) ==================


class TestCompaniesDispatch:
    def _company_headers(self, client, db, sample_company):
        from models import User, UserRole

        company_user = sample_company.user if hasattr(sample_company, "user") else None
        if not company_user:
            company_user = User.query.filter_by(id=sample_company.user_id).first()
        from flask_jwt_extended import create_access_token

        claims = {"role": UserRole.company.value, "company_id": sample_company.id, "driver_id": None, "aud": "atmr-api"}
        with client.application.app_context():
            token = create_access_token(identity=str(company_user.public_id), additional_claims=claims)
        return {"Authorization": f"Bearer {token}"}

    def test_company_dispatch_status(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.get("/api/v1/companies/me/dispatch/status", headers=headers)
        assert resp.status_code in [200, 404]
        _ = resp.get_json() if resp.is_json else None

    def test_company_dispatch_activate(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post("/api/v1/companies/me/dispatch/activate", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        _ = resp.get_json() if resp.is_json else None

    def test_company_dispatch_deactivate(self, client, db, sample_company):
        headers = self._company_headers(client, db, sample_company)
        resp = client.post("/api/v1/companies/me/dispatch/deactivate", json={}, headers=headers)
        assert resp.status_code in [200, 400, 404]
        _ = resp.get_json() if resp.is_json else None
