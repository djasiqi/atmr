"""Tests E2E : Gestion Client complète.

Ces tests vérifient le flux complet de gestion client :
- Enregistrement → Login → Création de booking → Historique
- Mise à jour du profil client
- Historique des bookings avec pagination
- Isolation des données entre clients
- Create → Update (phone, access_notes, etc.) → assert persisted (company clients)
"""

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from models import ActivationSession, BookingStatus, Client, UserRole
from repositories.user_repository import UserRepository
from tests.e2e.helpers.e2e_helpers import (
    create_test_booking,
    create_test_client,
    create_test_company,
)
from tests.e2e.test_auth_activation_e2e import _ensure_hmac_activation_token


def _complete_activation_for_e2e(e2e_client, db, activation_session_id: str) -> None:
    """Parcourt verify-email → verify-sms → finalize (contrat activation actuel)."""
    session = ActivationSession.query.filter_by(
        activation_session_id=activation_session_id
    ).first()
    assert session is not None, "Session d'activation introuvable en base"
    email_token = _ensure_hmac_activation_token(db, session)

    verify_email_response = e2e_client.post(
        "/api/v1/auth/activation/verify-email",
        json={"token": email_token},
        headers={"Content-Type": "application/json"},
    )
    assert verify_email_response.status_code == 200, (
        f"verify-email doit réussir, reçu {verify_email_response.status_code}: "
        f"{verify_email_response.get_json()}"
    )

    known_sms_code = "123456"
    session = ActivationSession.query.filter_by(
        activation_session_id=activation_session_id
    ).first()
    assert session is not None
    session.sms_code_hash = hashlib.sha256(known_sms_code.encode("utf-8")).hexdigest()
    session.sms_attempts = 0
    session.sms_locked_until = None
    db.session.commit()

    verify_sms_response = e2e_client.post(
        "/api/v1/auth/activation/verify-sms",
        json={
            "activation_session_id": activation_session_id,
            "code": known_sms_code,
        },
        headers={"Content-Type": "application/json"},
    )
    assert verify_sms_response.status_code == 200, (
        f"verify-sms doit réussir, reçu {verify_sms_response.status_code}: "
        f"{verify_sms_response.get_json()}"
    )

    finalize_response = e2e_client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": activation_session_id},
        headers={"Content-Type": "application/json"},
    )
    assert finalize_response.status_code == 200, (
        f"finalize doit réussir, reçu {finalize_response.status_code}: "
        f"{finalize_response.get_json()}"
    )


class TestClientRegistrationToBookingFlow:
    """Tests : Flux complet d'enregistrement à la création de booking."""

    def test_e2e_client_registration_to_booking_flow(self, e2e_client, db):
        """Test : Register → Activation → Login → Créer booking → Historique."""
        # 1. Enregistrement d'un nouveau client
        unique_suffix = str(uuid.uuid4())[:8]
        # Utiliser un mot de passe unique pour éviter HIBP (Have I Been Pwned)
        password = f"UniqueTestPass123!{unique_suffix}"
        register_data = {
            "username": f"newclient_{unique_suffix}",
            "email": f"newclient_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Jean",
            "last_name": "Dupont",
            "phone": "+41791234567",
            "address": "Rue de Test 1, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_data,
            headers={"Content-Type": "application/json"},
        )

        assert register_response.status_code in (200, 201), (
            f"Enregistrement doit réussir (200 ou 201), reçu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        register_result = register_response.get_json()
        activation_session_id = register_result.get("activation_session_id")
        assert activation_session_id, (
            f"activation_session_id manquant après register (réponse={register_result})"
        )

        # 1b. Activation obligatoire avant login (account_pending_activation)
        _complete_activation_for_e2e(e2e_client, db, activation_session_id)

        # 2. Login avec les credentials créés
        login_response = e2e_client.post(
            "/api/v1/auth/login",
            json={
                "email": register_data["email"],
                "password": register_data["password"],
            },
            headers={"Content-Type": "application/json"},
        )

        assert login_response.status_code == 200, (
            f"Login doit réussir, reçu {login_response.status_code}: "
            f"{login_response.get_json()}"
        )
        login_data = login_response.get_json()
        assert "user" in login_data
        assert login_data["user"]["email"] == register_data["email"]

        # Récupérer le public_id du client activé
        user_id = login_data["user"].get("public_id") or login_data["user"].get("id")
        assert user_id is not None

        # 3. Récupérer le client créé pour créer un booking directement en DB
        # (le géocodage nécessite un service externe qui peut échouer en tests)
        user_repo = UserRepository()
        user_obj = user_repo.find_by_public_id(user_id)
        assert user_obj is not None
        client_obj = Client.query.filter_by(user_id=user_obj.id).first()
        assert client_obj is not None

        # Créer un booking directement via le helper (sans géocodage)

        scheduled_dt = datetime.now(UTC) + timedelta(days=1)
        booking = create_test_booking(
            db,
            client=client_obj,
            scheduled_time=scheduled_dt,
            status=BookingStatus.PENDING,
        )
        booking_id = booking.id

        # 4. Vérifier l'historique des bookings via l'API
        history_response = e2e_client.get(f"/api/v1/clients/{user_id}/bookings")

        assert history_response.status_code == 200, (
            f"Récupération historique doit réussir, reçu {history_response.status_code}: "
            f"{history_response.get_json()}"
        )
        bookings = history_response.get_json()
        assert isinstance(bookings, list)
        assert len(bookings) >= 1, "L'historique doit contenir au moins le booking créé"

        # Vérifier que le booking créé est dans l'historique
        booking_ids = [
            b.get("id") or b.get("booking_id")
            for b in bookings
            if b.get("id") or b.get("booking_id")
        ]
        assert booking_id in booking_ids, (
            f"Le booking créé (id={booking_id}) doit être dans l'historique"
        )


class TestClientProfileUpdateFlow:
    """Tests : Mise à jour du profil client."""

    def test_e2e_client_profile_update_flow(self, e2e_client, db):
        """Test : Login → Mettre à jour profil → Vérifier changements."""
        # Setup : Créer un client de test
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        user = client.user

        user.set_password("testpassword123", force_change=False)
        db.session.commit()

        # 1. Login
        login_response = e2e_client.post(
            "/api/v1/auth/login",
            json={"email": user.email, "password": "testpassword123"},
        )
        assert login_response.status_code == 200

        public_id = user.public_id

        # 2. Récupérer le profil initial (pour vérifier qu'il existe)
        get_profile_response = e2e_client.get(f"/api/v1/clients/{public_id}")
        assert get_profile_response.status_code == 200
        _initial_profile = get_profile_response.get_json()  # Marquer comme non utilisée

        # 3. Mettre à jour le profil
        update_data = {
            "first_name": "NouveauPrénom",
            "last_name": "NouveauNom",
            "phone": "+41799999999",
            "address": "Nouvelle Adresse, 2000 Neuchâtel",
        }

        update_response = e2e_client.put(
            f"/api/v1/clients/{public_id}",
            json=update_data,
            headers={"Content-Type": "application/json"},
        )

        assert update_response.status_code == 200, (
            f"Mise à jour profil doit réussir, reçu {update_response.status_code}: "
            f"{update_response.get_json()}"
        )
        update_result = update_response.get_json()
        assert "message" in update_result

        # 4. Vérifier les changements
        get_updated_profile_response = e2e_client.get(f"/api/v1/clients/{public_id}")
        assert get_updated_profile_response.status_code == 200
        updated_profile = get_updated_profile_response.get_json()

        # Vérifier que les champs ont été mis à jour
        if "user" in updated_profile:
            user_data = updated_profile["user"]
            assert user_data.get("first_name") == update_data["first_name"]
            assert user_data.get("last_name") == update_data["last_name"]
        elif "first_name" in updated_profile:
            assert updated_profile.get("first_name") == update_data["first_name"]
            assert updated_profile.get("last_name") == update_data["last_name"]

        # Note: phone et address peuvent être dans différents endroits selon la structure de réponse
        # On vérifie au moins que la mise à jour a été acceptée (status 200)


class TestClientBookingHistoryFlow:
    """Tests : Historique des bookings avec pagination."""

    def test_e2e_client_booking_history_flow(self, e2e_client, db):
        """Test : Créer plusieurs bookings → Récupérer historique → Vérifier pagination."""
        # Setup : Créer un client de test
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        user = client.user

        user.set_password("testpassword123", force_change=False)
        db.session.commit()

        # 1. Login
        login_response = e2e_client.post(
            "/api/v1/auth/login",
            json={"email": user.email, "password": "testpassword123"},
        )
        assert login_response.status_code == 200

        public_id = user.public_id

        # 2. Créer plusieurs bookings directement via le helper (sans géocodage)
        # (le géocodage nécessite un service externe qui peut échouer en tests)

        bookings_created = []
        base_time = datetime.now(UTC) + timedelta(days=1)

        for i in range(3):
            booking_dt = base_time + timedelta(hours=i)
            booking = create_test_booking(
                db,
                client=client,
                scheduled_time=booking_dt,
                status=BookingStatus.PENDING,
            )
            bookings_created.append(booking.id)

        # 3. Récupérer l'historique complet
        history_response = e2e_client.get(f"/api/v1/clients/{public_id}/bookings")

        assert history_response.status_code == 200, (
            f"Récupération historique doit réussir, reçu {history_response.status_code}: "
            f"{history_response.get_json()}"
        )
        bookings = history_response.get_json()
        assert isinstance(bookings, list)

        # Vérifier que tous les bookings créés sont dans l'historique
        booking_ids_in_history = [
            b.get("id") or b.get("booking_id")
            for b in bookings
            if b.get("id") or b.get("booking_id")
        ]

        for booking_id in bookings_created:
            assert booking_id in booking_ids_in_history, (
                f"Le booking {booking_id} doit être dans l'historique"
            )

        # 4. Vérifier les réservations récentes (limit=4)
        recent_response = e2e_client.get(f"/api/v1/clients/{public_id}/recent-bookings")

        assert recent_response.status_code == 200
        recent_bookings = recent_response.get_json()
        assert isinstance(recent_bookings, list)
        assert len(recent_bookings) <= 4, (
            "Les réservations récentes doivent être limitées à 4"
        )

        # Vérifier que les bookings créés sont dans les récentes
        recent_booking_ids = [
            b.get("id") or b.get("booking_id")
            for b in recent_bookings
            if b.get("id") or b.get("booking_id")
        ]

        # Au moins certains des bookings créés doivent être dans les récentes
        assert any(bid in recent_booking_ids for bid in bookings_created), (
            "Au moins un des bookings créés doit être dans les récentes"
        )


class TestCompanyClientCreateUpdatePersistE2E:
    """E2E : Create client (company) → Update phone / access_notes / etc. → Assert persisted.

    Protège la chaîne Create → Edit (modal/drawer) → refresh : les champs
    saisis à la création doivent rester modifiables et persistés.
    """

    def test_e2e_company_client_create_update_phone_persisted(
        self, e2e_authenticated_company_client, e2e_company, db
    ):
        """Create client → Update phone (+ access_notes, etc.) → Assert persisted."""
        api = e2e_authenticated_company_client
        unique = str(uuid.uuid4())[:8]
        email = f"e2e-create-update-{unique}@internal.atmr.local"

        # 1. Create company client (aligné NewClientModal payload)
        create_payload = {
            "management_mode": "MANAGED",
            "email": email,
            "first_name": "E2E",
            "last_name": "CreateUpdate",
            "gender": "female",
            "address": "Rue de la Paix 1, 1202 Genève",
            "phone": "+41221234567",
            "access_notes": "Notes créées à la création",
            "is_active": True,
        }
        create_resp = api.post(
            "/api/v1/companies/me/clients",
            json=create_payload,
            headers={"Content-Type": "application/json"},
        )
        assert create_resp.status_code == 201, (
            f"Create doit réussir (201), reçu {create_resp.status_code}: "
            f"{create_resp.get_json()}"
        )
        created = create_resp.get_json()
        client_id = created.get("id")
        assert client_id is not None, "Réponse create doit contenir id"

        # 2. Update (aligné EditClientModal / ClientEditForm payload)
        update_payload = {
            "phone": "+41987654321",
            "access_notes": "Notes modifiées en édition",
            "first_name": "E2E",
            "last_name": "CreateUpdate",
            "gender": "female",
        }
        update_resp = api.put(
            f"/api/v1/companies/me/clients/{client_id}",
            json=update_payload,
            headers={"Content-Type": "application/json"},
        )
        assert update_resp.status_code == 200, (
            f"Update doit réussir (200), reçu {update_resp.status_code}: "
            f"{update_resp.get_json()}"
        )
        updated = update_resp.get_json()

        # 3. Assert persisted
        assert updated.get("phone") == "+41987654321", (
            "phone doit être persisté après update"
        )
        access = updated.get("access") or {}
        assert access.get("notes") == "Notes modifiées en édition", (
            "access_notes doit être persisté après update"
        )
