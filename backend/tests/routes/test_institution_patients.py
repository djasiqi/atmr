# tests/routes/test_institution_patients.py
"""Tests pour les endpoints patients institutionnels.

Ce module teste:
- Création de patients (JWT et API Key)
- Liste et recherche de patients
- Détail et modification de patients
- Idempotence avec external_reference
"""

import uuid

import pytest
from flask_jwt_extended import create_access_token

from models import Institution, InstitutionPatient, User, UserRole
from models.enums import InstitutionRole
from models.institution_api_key import InstitutionApiKey, generate_api_key


class TestInstitutionPatientsCRUD:
    """Tests CRUD pour les patients institutionnels."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Patients Test"
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
        user.username = f"patient_admin_{unique_suffix}"
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
        """Crée une clé API avec scopes patients."""
        raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = sample_institution.id
        api_key.name = "Test Patients API Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["patients:read", "patients:write"])
        db.session.add(api_key)
        db.session.commit()
        api_key._raw_key = raw_key
        return api_key

    def test_create_patient_jwt(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: création d'un patient avec JWT."""
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Jean",
                "last_name": "Dupont",
                "dob": "1985-03-15",
                "gender": "HOMME",
                "phone": "+41791234567",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        patient = data["patient"]
        assert patient["first_name"] == "Jean"
        assert patient["last_name"] == "Dupont"
        assert patient["dob"] == "1985-03-15"
        assert "id" in patient
        assert "public_id" in patient

    def test_create_patient_api_key(
        self, client, db, sample_api_key, sample_institution
    ):
        """Test: création d'un patient avec API Key."""
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Marie",
                "last_name": "Martin",
                "external_reference": "PAT-001",
            },
            headers={"X-API-Key": sample_api_key._raw_key},
        )

        assert response.status_code == 201
        data = response.get_json()
        patient = data["patient"]
        assert patient["first_name"] == "Marie"
        assert patient["external_reference"] == "PAT-001"

    def test_create_patient_duplicate_external_reference(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: création avec external_reference dupliqué -> 409."""
        # Créer premier patient
        response1 = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Patient",
                "last_name": "Un",
                "external_reference": "DUP-001",
            },
            headers=admin_auth_headers,
        )
        assert response1.status_code == 201

        # Tenter de créer un second avec même external_reference
        response2 = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Patient",
                "last_name": "Deux",
                "external_reference": "DUP-001",
            },
            headers=admin_auth_headers,
        )
        assert response2.status_code == 409
        assert "existe déjà" in response2.get_json().get("error", "")

    def test_list_patients(self, client, db, admin_auth_headers, sample_institution):
        """Test: liste des patients."""
        # Créer quelques patients
        for i in range(3):
            patient = InstitutionPatient()
            patient.institution_id = sample_institution.id
            patient.first_name = f"Patient{i}"
            patient.last_name = f"Test{i}"
            patient.public_id = str(uuid.uuid4())
            db.session.add(patient)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/patients",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "patients" in data
        assert len(data["patients"]) >= 3
        assert "total" in data

    def test_list_patients_pagination_default_limit(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: sans per_page, la liste est plafonnée à 20 (défaut API)."""
        for i in range(25):
            patient = InstitutionPatient()
            patient.institution_id = sample_institution.id
            patient.first_name = f"Pag{i}"
            patient.last_name = f"Limit{i:02d}"
            patient.public_id = str(uuid.uuid4())
            db.session.add(patient)
        db.session.commit()

        default_resp = client.get(
            "/api/v1/institutions/patients",
            headers=admin_auth_headers,
        )
        assert default_resp.status_code == 200
        default_data = default_resp.get_json()
        assert default_data["total"] >= 25
        assert default_data["per_page"] == 20
        assert len(default_data["patients"]) == 20

        full_resp = client.get(
            "/api/v1/institutions/patients?per_page=500",
            headers=admin_auth_headers,
        )
        assert full_resp.status_code == 200
        full_data = full_resp.get_json()
        assert full_data["total"] >= 25
        assert len(full_data["patients"]) >= 25

    def test_list_patients_search(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: recherche de patients par nom."""
        # Créer un patient spécifique
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "UniqueFirstName"
        patient.last_name = "UniqueLastName"
        patient.public_id = str(uuid.uuid4())
        db.session.add(patient)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/patients?query=UniqueFirst",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert len(data["patients"]) >= 1
        assert any(p["first_name"] == "UniqueFirstName" for p in data["patients"])

    def test_list_patients_search_by_city_and_phone(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: recherche par ville et téléphone."""
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Marie"
        patient.last_name = "VillePhone"
        patient.city = "Carouge"
        patient.phone = "+41791112233"
        patient.public_id = str(uuid.uuid4())
        db.session.add(patient)
        db.session.commit()

        by_city = client.get(
            "/api/v1/institutions/patients?query=Carouge",
            headers=admin_auth_headers,
        )
        assert by_city.status_code == 200
        city_data = by_city.get_json()
        assert any(p["last_name"] == "VillePhone" for p in city_data["patients"])

        by_phone = client.get(
            "/api/v1/institutions/patients?query=41791112233",
            headers=admin_auth_headers,
        )
        assert by_phone.status_code == 200
        phone_data = by_phone.get_json()
        assert any(p["last_name"] == "VillePhone" for p in phone_data["patients"])

    def test_get_patient_by_id(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: récupération d'un patient par ID."""
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "GetById"
        patient.last_name = "Test"
        patient.public_id = str(uuid.uuid4())
        db.session.add(patient)
        db.session.commit()

        response = client.get(
            f"/api/v1/institutions/patients/{patient.id}",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["first_name"] == "GetById"

    def test_get_patient_by_external_reference(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """Test: récupération d'un patient par référence externe."""
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "ByRef"
        patient.last_name = "Test"
        patient.external_reference = "REF-UNIQUE-123"
        patient.public_id = str(uuid.uuid4())
        db.session.add(patient)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/patients/by-reference/REF-UNIQUE-123",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["external_reference"] == "REF-UNIQUE-123"

    def test_update_patient(self, client, db, admin_auth_headers, sample_institution):
        """Test: mise à jour d'un patient."""
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Original"
        patient.last_name = "Name"
        patient.public_id = str(uuid.uuid4())
        db.session.add(patient)
        db.session.commit()

        response = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"first_name": "Updated"},
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["first_name"] == "Updated"
        assert data["last_name"] == "Name"  # Non modifié

    def test_patient_not_found(self, client, db, admin_auth_headers):
        """Test: patient non trouvé -> 404."""
        response = client.get(
            "/api/v1/institutions/patients/99999",
            headers=admin_auth_headers,
        )

        assert response.status_code == 404

    def test_patient_no_auth(self, client, db):
        """Test: accès sans authentification -> 401."""
        response = client.get("/api/v1/institutions/patients")

        assert response.status_code == 401
