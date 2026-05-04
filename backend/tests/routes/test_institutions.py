# tests/routes/test_institutions.py
"""Tests pour les routes du portail institutionnel.

Ce module teste:
- GET /api/institutions/me (endpoint probe)
- Isolation entre tenants (institution vs company)
- Validation des claims JWT institution_id
"""

import uuid

import pytest
from flask_jwt_extended import create_access_token

from models import Institution, User, UserRole
from models.enums import InstitutionRole


class TestInstitutionMe:
    """Tests pour l'endpoint GET /api/institutions/me."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique du Test"
        institution.institution_type = "clinic"
        institution.address = "Rue de la Clinique 1, 1000 Lausanne"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        db.session.refresh(institution)
        return institution

    @pytest.fixture
    def sample_institution_user(self, db, sample_institution):
        """Crée un utilisateur institution de test."""
        unique_suffix = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"institution_user_{unique_suffix}"
        user.email = f"inst-{unique_suffix}@example.com"
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
    def institution_auth_headers(
        self, client, sample_institution_user, sample_institution
    ):
        """Génère un token JWT valide pour un utilisateur institution."""
        claims = {
            "role": sample_institution_user.role.value,
            "institution_id": sample_institution.id,
            "institution_role": sample_institution_user.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(sample_institution_user.public_id),
                additional_claims=claims,
            )
        return {"Authorization": f"Bearer {token}"}

    def test_institution_me_success(
        self,
        client,
        db,
        sample_institution,
        sample_institution_user,
        institution_auth_headers,
    ):
        """Test: un user institution peut accéder à /api/institutions/me."""
        response = client.get(
            "/api/v1/institutions/me",
            headers=institution_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()

        # Vérifier les champs retournés
        assert data["id"] == sample_institution.id
        assert data["public_id"] == sample_institution.public_id
        assert data["name"] == "Clinique du Test"
        assert data["institution_type"] == "clinic"
        assert data["institution_role"] == InstitutionRole.ADMIN.value

        # Vérifier les informations utilisateur
        assert "user" in data
        assert data["user"]["id"] == sample_institution_user.id
        assert data["user"]["public_id"] == sample_institution_user.public_id

    def test_institution_me_company_user_forbidden(
        self, client, db, sample_user, auth_headers
    ):
        """Test: un user company ne peut PAS accéder à /api/institutions/me."""
        response = client.get(
            "/api/v1/institutions/me",
            headers=auth_headers,
        )

        # Le company user devrait recevoir 403 (accès refusé)
        assert response.status_code == 403

    def test_institution_me_no_institution_claim(self, client, db):
        """Test: un user institution sans institution_id claim -> 403."""
        # Créer un user institution SANS institution_id dans les claims
        user = User()
        user.username = f"no_claim_user_{uuid.uuid4().hex[:8]}"
        user.email = f"noclaim-{uuid.uuid4().hex[:8]}@example.com"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        # PAS de institution_id assigné
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()

        # Créer un token SANS institution_id claim
        claims = {
            "role": UserRole.INSTITUTION.value,
            # PAS de institution_id
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
            )
        headers = {"Authorization": f"Bearer {token}"}

        response = client.get(
            "/api/v1/institutions/me",
            headers=headers,
        )

        # Devrait recevoir 403 car pas de institution_id claim
        assert response.status_code == 403

    def test_institution_me_no_auth(self, client):
        """Test: accès sans token -> 401."""
        response = client.get("/api/v1/institutions/me")

        assert response.status_code == 401

    def test_institution_me_admin_can_access(
        self, client, db, sample_admin_user, admin_headers, sample_institution
    ):
        """Test: un admin ne peut PAS accéder directement (pas le bon rôle).

        Les admins doivent avoir un token avec role=institution pour accéder.
        """
        response = client.get(
            "/api/v1/institutions/me",
            headers=admin_headers,
        )

        # Admin n'a pas le rôle institution, donc 403
        assert response.status_code == 403


class TestInstitutionModel:
    """Tests pour le modèle Institution."""

    def test_create_institution(self, db):
        """Test: création d'une institution."""
        institution = Institution()
        institution.name = "EMS Les Tilleuls"
        institution.institution_type = "ems"
        institution.address = "Avenue des Tilleuls 10, 1005 Lausanne"
        institution.contact_email = "contact@ems-tilleuls.ch"
        institution.contact_phone = "0213456789"
        institution.public_id = str(uuid.uuid4())

        db.session.add(institution)
        db.session.flush()

        assert institution.id is not None
        assert institution.name == "EMS Les Tilleuls"
        assert institution.institution_type == "ems"
        assert institution.created_at is not None

    def test_institution_serialize(self, db):
        """Test: sérialisation d'une institution."""
        institution = Institution()
        institution.name = "Hôpital du Test"
        institution.institution_type = "hospital"
        institution.public_id = str(uuid.uuid4())

        db.session.add(institution)
        db.session.flush()

        serialized = institution.serialize

        assert "id" in serialized
        assert serialized["name"] == "Hôpital du Test"
        assert serialized["institution_type"] == "hospital"
        assert "created_at" in serialized


class TestInstitutionUserRelation:
    """Tests pour la relation User <-> Institution."""

    def test_user_with_institution(self, db):
        """Test: un user peut être associé à une institution."""
        # Créer institution
        institution = Institution()
        institution.name = "IMAD Genève"
        institution.institution_type = "imad"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()

        # Créer user institution
        user = User()
        user.username = f"imad_user_{uuid.uuid4().hex[:8]}"
        user.email = f"user-{uuid.uuid4().hex[:8]}@imad.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()

        # Vérifier la relation
        assert user.institution_id == institution.id
        assert user.institution_role == InstitutionRole.REQUESTER.value
        assert user.institution.name == "IMAD Genève"

    def test_institution_has_multiple_users(self, db):
        """Test: une institution peut avoir plusieurs utilisateurs."""
        institution = Institution()
        institution.name = "Clinique Multi-Users"
        institution.institution_type = "clinic"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()

        # Créer plusieurs utilisateurs
        users = []
        for i, role in enumerate(
            [InstitutionRole.ADMIN, InstitutionRole.REQUESTER, InstitutionRole.READER]
        ):
            user = User()
            user.username = f"clinic_user_{i}_{uuid.uuid4().hex[:8]}"
            user.email = f"user{i}-{uuid.uuid4().hex[:8]}@clinic.ch"
            user.role = UserRole.INSTITUTION
            user.public_id = str(uuid.uuid4())
            user.institution_id = institution.id
            user.institution_role = role.value
            user.set_password("password123", force_change=False)
            db.session.add(user)
            users.append(user)

        db.session.flush()

        # Vérifier que l'institution a bien plusieurs users
        assert len(institution.users) == 3
        roles = {u.institution_role for u in institution.users}
        assert InstitutionRole.ADMIN.value in roles
        assert InstitutionRole.REQUESTER.value in roles
        assert InstitutionRole.READER.value in roles


class TestInstitutionRoleValidation:
    """Tests pour la validation des rôles institution."""

    def test_valid_institution_roles(self, db):
        """Test: tous les rôles institution valides sont acceptés."""
        institution = Institution()
        institution.name = "Test Institution"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()

        for role in InstitutionRole:
            user = User()
            user.username = f"user_{role.value}_{uuid.uuid4().hex[:8]}"
            user.email = f"{role.value}-{uuid.uuid4().hex[:8]}@test.ch"
            user.role = UserRole.INSTITUTION
            user.public_id = str(uuid.uuid4())
            user.institution_id = institution.id
            user.institution_role = role.value
            user.set_password("password123", force_change=False)
            db.session.add(user)

        db.session.flush()  # Ne devrait pas lever d'exception

    def test_invalid_institution_role(self, db):
        """Test: un rôle institution invalide lève une erreur."""
        institution = Institution()
        institution.name = "Test Institution"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()

        user = User()
        user.username = f"invalid_role_user_{uuid.uuid4().hex[:8]}"
        user.email = f"invalid-{uuid.uuid4().hex[:8]}@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id

        with pytest.raises(ValueError, match="Invalid institution_role"):
            user.institution_role = "invalid_role"


class TestInstitutionSettingsTimezone:
    """Tests pour la validation timezone dans PUT /institutions/settings."""

    @pytest.fixture
    def institution(self, db):
        inst = Institution()
        inst.name = "Clinique TZ Test"
        inst.institution_type = "clinic"
        inst.public_id = str(uuid.uuid4())
        db.session.add(inst)
        db.session.flush()
        db.session.refresh(inst)
        return inst

    @pytest.fixture
    def admin_user(self, db, institution):
        user = User()
        user.username = f"tz_admin_{uuid.uuid4().hex[:8]}"
        user.email = f"tz-{uuid.uuid4().hex[:8]}@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.ADMIN.value
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()
        db.session.refresh(user)
        return user

    @pytest.fixture
    def headers(self, client, admin_user, institution):
        claims = {
            "role": admin_user.role.value,
            "institution_id": institution.id,
            "institution_role": admin_user.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(admin_user.public_id),
                additional_claims=claims,
            )
        return {"Authorization": f"Bearer {token}"}

    def test_put_settings_invalid_timezone_returns_400(
        self, client, db, institution, admin_user, headers
    ):
        """PUT /settings avec timezone invalide doit retourner 400."""
        response = client.put(
            "/api/v1/institutions/settings",
            json={"timezone": "Mars/Olympus_Mons"},
            headers=headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        # Le message d'erreur doit mentionner la timezone
        errors = data.get("errors") or data.get("error", "")
        assert "timezone" in str(errors).lower() or "invalide" in str(errors).lower()

    def test_put_settings_valid_timezone_accepted(
        self, client, db, institution, admin_user, headers
    ):
        """PUT /settings avec timezone IANA valide doit retourner 200."""
        response = client.put(
            "/api/v1/institutions/settings",
            json={"timezone": "America/New_York"},
            headers=headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["settings"]["timezone"] == "America/New_York"

    def test_put_settings_common_timezone_accepted(
        self, client, db, institution, admin_user, headers
    ):
        """PUT /settings avec timezone commune (Europe/Zurich) doit fonctionner."""
        response = client.put(
            "/api/v1/institutions/settings",
            json={"timezone": "Europe/Zurich"},
            headers=headers,
        )
        assert response.status_code == 200
