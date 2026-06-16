# tests/routes/test_institution_requests.py
# ruff: noqa: I001
"""Tests pour les endpoints demandes de transport institutionnelles.

Ce module teste:
- Création de demandes (JWT et API Key)
- Liste et filtrage de demandes
- Modification de demandes
- Envoi et annulation de demandes
- Idempotence avec external_reference
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from flask_jwt_extended import create_access_token

from models import Institution, InstitutionPatient, TransportRequest, User, UserRole
from models.enums import InstitutionRole, RequestStatus
from models.institution_api_key import InstitutionApiKey, generate_api_key


class TestTransportRequestsCRUD:
    """Tests CRUD pour les demandes de transport."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Requests Test"
        institution.institution_type = "clinic"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        db.session.refresh(institution)
        return institution

    @pytest.fixture
    def sample_institution_admin(self, db, sample_institution):
        """Crée un utilisateur admin institution."""
        unique_suffix = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"request_admin_{unique_suffix}"
        user.email = f"admin-{unique_suffix}@clinic.test"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = sample_institution.id
        user.institution_role = InstitutionRole.ADMIN.value
        user.set_password("password123", force_change=False)

        db.session.add(user)
        db.session.flush()
        db.session.refresh(user)
        return user

    @pytest.fixture
    def admin_auth_headers(self, client, sample_institution_admin, sample_institution):
        """Génère un token JWT pour admin institution."""
        claims = {
            "role": sample_institution_admin.role.value,
            "institution_id": sample_institution.id,
            "institution_role": sample_institution_admin.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(sample_institution_admin.public_id),
                additional_claims=claims,
            )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def sample_api_key(self, db, sample_institution):
        """Crée une clé API avec scopes requests."""
        raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = sample_institution.id
        api_key.name = "Test Requests API Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["requests:read", "requests:write", "requests:cancel"])
        db.session.add(api_key)
        db.session.commit()
        api_key._raw_key = raw_key
        return api_key

    @pytest.fixture
    def sample_patient(self, db, sample_institution):
        """Crée un patient de test."""
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Test"
        patient.last_name = "Patient"
        patient.public_id = str(uuid.uuid4())
        patient.external_reference = "PAT-TEST-001"
        db.session.add(patient)
        db.session.flush()
        return patient

    def _get_scheduled_time(self, hours_ahead: int = 24) -> str:
        """Génère un scheduled_time ISO8601 dans le futur."""
        dt = datetime.now(UTC) + timedelta(hours=hours_ahead)
        return dt.isoformat()

    def _future_scheduled_datetime(self, hours_ahead: int = 24) -> datetime:
        """Datetime futur pour fixtures TransportRequest (mission_date requis)."""
        return datetime.now(UTC) + timedelta(hours=hours_ahead)

    def _populate_request_schedule(
        self, req: TransportRequest, scheduled: datetime
    ) -> None:
        """Renseigne mission_date + confirmation départ (requis pour send)."""
        req.scheduled_time = scheduled
        req.mission_date = scheduled.date()
        req.pickup_time_confirmed = True
        if not getattr(req, "billing_intent", None):
            req.billing_intent = "patient"

    def test_create_request_jwt(
        self, client, db, admin_auth_headers, sample_institution, sample_patient
    ):
        """Test: création d'une demande avec JWT."""
        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": "REQ-001",
                "patient_id": sample_patient.id,
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "Clinique ABC, 1 rue Test",
                "dropoff_location": "Hôpital XYZ, 2 avenue Example",
                "billing_intent": "patient",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["external_reference"] == "REQ-001"
        assert data["status"] == "DRAFT"
        assert data["patient_id"] == sample_patient.id
        assert data["is_editable"] is True

    def test_create_request_api_key(
        self, client, db, sample_api_key, sample_institution
    ):
        """Test: création d'une demande avec API Key."""
        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": "REQ-API-001",
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "123 rue A",
                "dropoff_location": "456 rue B",
                "mission_type": "patient_transport",
            },
            headers={"X-API-Key": sample_api_key._raw_key},
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["external_reference"] == "REQ-API-001"

    def test_create_request_without_external_reference(
        self, client, db, admin_auth_headers
    ):
        """Test: création sans external_reference -> 201 (champ optionnel)."""
        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "123 rue A",
                "dropoff_location": "456 rue B",
                "mission_type": "patient_transport",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert "external_reference" in data
        assert data["external_reference"] is None

    def test_create_request_material_delivery(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: création d'une demande de livraison matériel."""
        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": "DEL-001",
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "Pharmacie",
                "dropoff_location": "Clinique",
                "mission_type": "material_delivery",
                "delivery_description": "Médicaments urgents",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["mission_type"] == "material_delivery"
        assert data["delivery_description"] == "Médicaments urgents"

    def test_create_request_material_delivery_missing_description(
        self, client, db, admin_auth_headers
    ):
        """Test: livraison sans description -> 400."""
        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": "DEL-002",
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "A",
                "dropoff_location": "B",
                "mission_type": "material_delivery",
                # delivery_description manquant
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 400

    def test_create_request_duplicate_external_reference(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: création avec external_reference dupliqué -> 409."""
        # Créer première demande
        response1 = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": "DUP-REQ-001",
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "A",
                "dropoff_location": "B",
            },
            headers=admin_auth_headers,
        )
        assert response1.status_code == 201

        # Tenter de créer une seconde avec même référence
        response2 = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": "DUP-REQ-001",
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "C",
                "dropoff_location": "D",
            },
            headers=admin_auth_headers,
        )
        assert response2.status_code == 409
        data = response2.get_json()
        assert "existe déjà" in data.get("error", "")
        assert "existing_request_id" in data

    def test_list_requests(self, client, db, admin_auth_headers, sample_institution):
        """Test: liste des demandes."""
        # Créer quelques demandes
        for i in range(3):
            req = TransportRequest()
            req.institution_id = sample_institution.id
            req.external_reference = f"LIST-{i}-{uuid.uuid4().hex[:8]}"
            req.pickup_location = "A"
            req.dropoff_location = "B"
            scheduled = self._future_scheduled_datetime(hours_ahead=i + 1)
            self._populate_request_schedule(req, scheduled)
            req.public_id = str(uuid.uuid4())
            db.session.add(req)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/requests",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "requests" in data
        assert len(data["requests"]) >= 3
        assert "total" in data

    def test_list_requests_filter_status(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: filtrage par statut."""
        # Créer une demande SENT
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"SENT-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.status = RequestStatus.SENT.value
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/requests?status=SENT",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        # Tous les résultats doivent être SENT
        for r in data["requests"]:
            assert r["status"] == "SENT"

    def test_get_request_by_id(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: récupération d'une demande par ID."""
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"GET-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.get(
            f"/api/v1/institutions/requests/{req.id}",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["external_reference"] == req.external_reference

    def test_get_request_by_external_reference(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: récupération par référence externe."""
        ext_ref = f"BYREF-{uuid.uuid4().hex[:8]}"
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = ext_ref
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.get(
            f"/api/v1/institutions/requests/by-reference/{ext_ref}",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["external_reference"] == ext_ref

    def test_update_request_draft(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: modification d'une demande DRAFT."""
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"UPDATE-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "Original"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.status = RequestStatus.DRAFT.value
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.put(
            f"/api/v1/institutions/requests/{req.id}",
            json={"pickup_location": "Updated Location"},
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["pickup_location"] == "Updated Location"

    def test_update_request_cancelled_fails(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: modification d'une demande CANCELLED -> 400."""
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"CANCELLED-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.status = RequestStatus.CANCELLED.value
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.put(
            f"/api/v1/institutions/requests/{req.id}",
            json={"pickup_location": "New"},
            headers=admin_auth_headers,
        )

        assert response.status_code == 400
        assert "non modifiable" in response.get_json().get("error", "")

    def test_send_request(self, client, db, admin_auth_headers, sample_institution):
        """Test: envoi d'une demande DRAFT -> SENT."""
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"SEND-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.status = RequestStatus.DRAFT.value
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{req.id}/send",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "SENT"
        assert data["sent_at"] is not None

    def test_send_request_already_sent_fails(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: renvoi d'une demande déjà SENT relance ou idempotent (200)."""
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"ALREADY-SENT-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.status = RequestStatus.SENT.value
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{req.id}/send",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "SENT"

    def test_cancel_request_draft(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: annulation d'une demande DRAFT."""
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"CANCEL-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.status = RequestStatus.DRAFT.value
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{req.id}/cancel",
            json={"reason": "Test annulation"},
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "CANCELLED"
        assert data["cancelled_at"] is not None

    def test_cancel_request_converted_fails(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: annulation d'une demande CONVERTED -> 400."""
        req = TransportRequest()
        req.institution_id = sample_institution.id
        req.external_reference = f"CONVERTED-{uuid.uuid4().hex[:8]}"
        req.pickup_location = "A"
        req.dropoff_location = "B"
        scheduled = self._future_scheduled_datetime()
        self._populate_request_schedule(req, scheduled)
        req.status = RequestStatus.CONVERTED.value
        req.public_id = str(uuid.uuid4())
        db.session.add(req)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{req.id}/cancel",
            headers=admin_auth_headers,
        )

        assert response.status_code == 409
        assert "convertie" in response.get_json().get("error", "").lower()

    def test_request_not_found(self, client, db, admin_auth_headers):
        """Test: demande non trouvée -> 404."""
        response = client.get(
            "/api/v1/institutions/requests/99999",
            headers=admin_auth_headers,
        )

        assert response.status_code == 404

    def test_request_no_auth(self, client, db):
        """Test: accès sans authentification -> 401."""
        response = client.get("/api/v1/institutions/requests")

        assert response.status_code == 401

    def test_request_with_patient_external_reference(
        self, client, db, admin_auth_headers, sample_institution, sample_patient
    ):
        """Test: création avec patient_external_reference."""
        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": f"REQ-PAT-REF-{uuid.uuid4().hex[:8]}",
                "patient_external_reference": sample_patient.external_reference,
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "A",
                "dropoff_location": "B",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["patient_id"] == sample_patient.id

    def test_request_with_mobility_info(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: création avec informations de mobilité."""
        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": f"REQ-MOB-{uuid.uuid4().hex[:8]}",
                "scheduled_time": self._get_scheduled_time(),
                "pickup_location": "A",
                "dropoff_location": "B",
                "mobility": {
                    "wheelchair": True,
                    "needs_assistance": True,
                },
                "contact_on_site": {
                    "name": "Dr. Martin",
                    "phone": "+41791234567",
                },
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["mobility"]["wheelchair"] is True
        assert data["contact_on_site"]["name"] == "Dr. Martin"


class TestTransportRequestsForcePasswordChange:
    """Les routes JWT institution bloquent le MDP temporaire non changé."""

    @pytest.fixture
    def sample_institution(self, db):
        institution = Institution()
        institution.name = "Clinique Force Password"
        institution.institution_type = "clinic"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        db.session.refresh(institution)
        return institution

    def test_create_request_blocked_until_password_changed(
        self, client, db, sample_institution
    ):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"requester.{uid}"
        user.email = None
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = sample_institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "active"
        user.authentication_method = "username"
        user.password_expires_at = datetime.now(UTC) + timedelta(days=14)
        user.set_password("TempPass123!Xy", force_change=True)
        db.session.add(user)
        db.session.commit()

        claims = {
            "role": user.role.value,
            "institution_id": sample_institution.id,
            "institution_role": user.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
            )
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post(
            "/api/v1/institutions/requests",
            json={
                "external_reference": f"REQ-FPC-{uuid.uuid4().hex[:8]}",
                "scheduled_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                "pickup_location": "A",
                "dropoff_location": "B",
            },
            headers=headers,
        )

        assert response.status_code == 403
        data = response.get_json()
        assert data["error"] == "password_change_required"
