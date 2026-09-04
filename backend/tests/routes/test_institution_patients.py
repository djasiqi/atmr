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

from models import Institution, InstitutionPatient, User, UserRole
from models.enums import InstitutionRole
from models.institution_api_key import InstitutionApiKey, generate_api_key
from tests.helpers.institution_auth import institution_bearer_headers

COMPLETE_DOMICILE = {
    "address": "12 rue du Lac",
    "postal_code": "1200",
    "city": "Genève",
}


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
    def admin_auth_headers(self, db, sample_institution_admin, sample_institution):
        """Génère un token JWT pour admin institution."""
        return institution_bearer_headers(
            db,
            sample_institution_admin,
            sample_institution,
            institution_role=sample_institution_admin.institution_role,
        )

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
                **COMPLETE_DOMICILE,
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
                "dob": "1988-06-20",
                "gender": "FEMME",
                "external_reference": "PAT-001",
                **COMPLETE_DOMICILE,
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
                "dob": "1975-01-10",
                "gender": "HOMME",
                "external_reference": "DUP-001",
                **COMPLETE_DOMICILE,
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
                "dob": "1976-02-11",
                "gender": "FEMME",
                "external_reference": "DUP-001",
                **COMPLETE_DOMICILE,
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

    def test_create_patient_missing_identity_400(self, client, db, admin_auth_headers):
        """PATIENT-IDENTITY-01 : civilité + DOB obligatoires à la création."""
        response = client.post(
            "/api/v1/institutions/patients",
            json={"first_name": "Sans", "last_name": "Identite"},
            headers=admin_auth_headers,
        )
        assert response.status_code == 400

    def test_create_patient_minor_without_confirmation_422(
        self, client, db, admin_auth_headers
    ):
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        minor = adult_dob_cutoff(date.today()) + timedelta(days=1)
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Mineur",
                "last_name": "Test",
                "gender": "HOMME",
                "dob": minor.isoformat(),
                **COMPLETE_DOMICILE,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 422
        body = response.get_json() or {}
        assert body.get("code") == "MINOR_DOB_CONFIRMATION_REQUIRED"

    def test_create_patient_minor_confirmed_201(self, client, db, admin_auth_headers):
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        minor = adult_dob_cutoff(date.today()) + timedelta(days=1)
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Mineur",
                "last_name": "Ok",
                "gender": "FEMME",
                "dob": minor.isoformat(),
                "minor_dob_confirmed": True,
                **COMPLETE_DOMICILE,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 201, response.get_json()
        assert response.get_json()["patient"]["dob"] == minor.isoformat()

    def test_force_create_minor_without_confirmation_still_422(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """force_create ≠ bypass métier : mineur sans confirmation → 422."""
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        minor = adult_dob_cutoff(date.today()) + timedelta(days=1)
        # Doublon existant
        existing = InstitutionPatient()
        existing.institution_id = sample_institution.id
        existing.first_name = "Julie"
        existing.last_name = "Dupont"
        existing.public_id = str(uuid.uuid4())
        existing.dob = minor
        existing.gender = "FEMME"
        db.session.add(existing)
        db.session.commit()

        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Julie",
                "last_name": "Dupont",
                "gender": "FEMME",
                "dob": minor.isoformat(),
                "force_create": True,
                **COMPLETE_DOMICILE,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 422
        assert response.get_json().get("code") == "MINOR_DOB_CONFIRMATION_REQUIRED"

    def test_force_create_minor_with_confirmation_201(
        self, client, db, admin_auth_headers, sample_institution
    ):
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        minor = adult_dob_cutoff(date.today()) + timedelta(days=1)
        existing = InstitutionPatient()
        existing.institution_id = sample_institution.id
        existing.first_name = "Julie"
        existing.last_name = "Force"
        existing.public_id = str(uuid.uuid4())
        existing.dob = minor
        existing.gender = "FEMME"
        db.session.add(existing)
        db.session.commit()

        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Julie",
                "last_name": "Force",
                "gender": "FEMME",
                "dob": minor.isoformat(),
                "force_create": True,
                "minor_dob_confirmed": True,
                **COMPLETE_DOMICILE,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 201, response.get_json()

    def test_future_dob_with_minor_confirmed_still_rejected(
        self, client, db, admin_auth_headers
    ):
        from datetime import date, timedelta

        future = date.today() + timedelta(days=10)
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Futur",
                "last_name": "Hack",
                "gender": "HOMME",
                "dob": future.isoformat(),
                "minor_dob_confirmed": True,
                **COMPLETE_DOMICILE,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 400

    def test_adult_with_minor_confirmed_flag_ignored(
        self, client, db, admin_auth_headers
    ):
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Adulte",
                "last_name": "Flag",
                "gender": "HOMME",
                "dob": "1985-03-15",
                "minor_dob_confirmed": True,
                **COMPLETE_DOMICILE,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 201, response.get_json()
        assert response.get_json()["patient"]["dob"] == "1985-03-15"

    def test_update_adult_to_minor_requires_confirmation(
        self, client, db, admin_auth_headers, sample_institution
    ):
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Adult"
        patient.last_name = "Update"
        patient.public_id = str(uuid.uuid4())
        patient.dob = date(1980, 1, 1)
        patient.gender = "HOMME"
        db.session.add(patient)
        db.session.commit()

        minor = adult_dob_cutoff(date.today()) + timedelta(days=1)
        denied = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"dob": minor.isoformat()},
            headers=admin_auth_headers,
        )
        assert denied.status_code == 422
        assert denied.get_json().get("code") == "MINOR_DOB_CONFIRMATION_REQUIRED"

        ok = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"dob": minor.isoformat(), "minor_dob_confirmed": True},
            headers=admin_auth_headers,
        )
        assert ok.status_code == 200, ok.get_json()

    def test_update_minor_phone_without_reconfirm(
        self, client, db, admin_auth_headers, sample_institution
    ):
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        minor = adult_dob_cutoff(date.today()) + timedelta(days=1)
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Mineur"
        patient.last_name = "Phone"
        patient.public_id = str(uuid.uuid4())
        patient.dob = minor
        patient.gender = "HOMME"
        patient.phone = "+41791111111"
        db.session.add(patient)
        db.session.commit()

        response = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"phone": "+41792222222"},
            headers=admin_auth_headers,
        )
        assert response.status_code == 200, response.get_json()
        assert response.get_json()["phone"] == "+41792222222"

    def test_update_minor_to_other_minor_requires_confirmation(
        self, client, db, admin_auth_headers, sample_institution
    ):
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        m1 = adult_dob_cutoff(date.today()) + timedelta(days=1)
        m2 = adult_dob_cutoff(date.today()) + timedelta(days=40)
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Mineur"
        patient.last_name = "Change"
        patient.public_id = str(uuid.uuid4())
        patient.dob = m1
        patient.gender = "FEMME"
        db.session.add(patient)
        db.session.commit()

        denied = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"dob": m2.isoformat()},
            headers=admin_auth_headers,
        )
        assert denied.status_code == 422
        assert denied.get_json().get("code") == "MINOR_DOB_CONFIRMATION_REQUIRED"

        ok = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"dob": m2.isoformat(), "minor_dob_confirmed": True},
            headers=admin_auth_headers,
        )
        assert ok.status_code == 200, ok.get_json()
        assert ok.get_json()["dob"] == m2.isoformat()

    def test_update_minor_gender_without_reconfirm(
        self, client, db, admin_auth_headers, sample_institution
    ):
        from datetime import date, timedelta

        from application.institutions.patient_identity_rules import adult_dob_cutoff

        minor = adult_dob_cutoff(date.today()) + timedelta(days=1)
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Mineur"
        patient.last_name = "Gender"
        patient.public_id = str(uuid.uuid4())
        patient.dob = minor
        patient.gender = "HOMME"
        db.session.add(patient)
        db.session.commit()

        response = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"gender": "FEMME"},
            headers=admin_auth_headers,
        )
        assert response.status_code == 200, response.get_json()
        assert response.get_json()["gender"] == "FEMME"

    def test_create_without_address_400(self, client, db, admin_auth_headers):
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Sans",
                "last_name": "Adresse",
                "gender": "HOMME",
                "dob": "1985-03-15",
                "postal_code": "1200",
                "city": "Genève",
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 400
        details = (response.get_json() or {}).get("details") or {}
        assert "address" in details

    def test_create_without_postal_code_400(self, client, db, admin_auth_headers):
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Sans",
                "last_name": "Npa",
                "gender": "HOMME",
                "dob": "1985-03-15",
                "address": "12 rue du Lac",
                "city": "Genève",
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 400
        details = (response.get_json() or {}).get("details") or {}
        assert "postal_code" in details

    def test_create_without_city_400(self, client, db, admin_auth_headers):
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Sans",
                "last_name": "Ville",
                "gender": "HOMME",
                "dob": "1985-03-15",
                "address": "12 rue du Lac",
                "postal_code": "1200",
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 400
        details = (response.get_json() or {}).get("details") or {}
        assert "city" in details

    def test_create_complete_domicile_201(self, client, db, admin_auth_headers):
        response = client.post(
            "/api/v1/institutions/patients",
            json={
                "first_name": "Complet",
                "last_name": "Domicile",
                "gender": "FEMME",
                "dob": "1985-03-15",
                **COMPLETE_DOMICILE,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 201, response.get_json()
        patient = response.get_json()["patient"]
        assert patient["address"] == COMPLETE_DOMICILE["address"]
        assert patient["postal_code"] == COMPLETE_DOMICILE["postal_code"]
        assert patient["city"] == COMPLETE_DOMICILE["city"]

    def test_put_phone_legacy_incomplete_domicile_ok(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """PUT téléphone seul sur patient legacy sans adresse → autorisé."""
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Legacy"
        patient.last_name = "NoAddr"
        patient.public_id = str(uuid.uuid4())
        patient.dob = __import__("datetime").date(1980, 1, 1)
        patient.gender = "HOMME"
        patient.address = None
        patient.postal_code = None
        patient.city = None
        db.session.add(patient)
        db.session.commit()

        response = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"phone": "+41793334455"},
            headers=admin_auth_headers,
        )
        assert response.status_code == 200, response.get_json()
        assert response.get_json()["phone"] == "+41793334455"

    def test_put_partial_address_without_city_rejected(
        self, client, db, admin_auth_headers, sample_institution
    ):
        """PUT qui touche l'adresse → état final triplet cohérent."""
        patient = InstitutionPatient()
        patient.institution_id = sample_institution.id
        patient.first_name = "Legacy"
        patient.last_name = "Partial"
        patient.public_id = str(uuid.uuid4())
        patient.dob = __import__("datetime").date(1980, 1, 1)
        patient.gender = "HOMME"
        db.session.add(patient)
        db.session.commit()

        response = client.put(
            f"/api/v1/institutions/patients/{patient.id}",
            json={"address": "1 rue Test", "postal_code": "1200"},
            headers=admin_auth_headers,
        )
        assert response.status_code == 400
        details = (response.get_json() or {}).get("details") or {}
        assert "city" in details
