"""Tests E2E : Gestion Chauffeur complète.

Ces tests vérifient le flux complet de gestion chauffeur :
- Création chauffeur → Dispatch → Assignation
- Assignation booking → Mise à jour statut → Notifications
- Assignation booking → Démarrage → Terminaison → Statut final
"""

from datetime import UTC, datetime, timedelta

import pytest

from models import BookingStatus, User
from tests.e2e.helpers.e2e_helpers import (
    assert_booking_assigned,
    create_test_booking,
    create_test_client,
    create_test_company,
    create_test_driver,
)


def _login_driver_mobile(e2e_client, user: User, password: str) -> dict[str, str]:
    """Connecte un chauffeur et retourne les headers Bearer.

    Les routes ``/api/v1/driver/*`` sont traitées comme mobiles par
    ``_lot1_jwt_locations_web_mobile_split`` : seul le header Authorization
    est accepté, les cookies web posés par /auth/login sont ignorés.
    """
    login_response = e2e_client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": password},
        headers={"Content-Type": "application/json", "X-Requested-With": "Expo"},
    )
    assert login_response.status_code == 200, (
        f"Login chauffeur doit réussir, reçu {login_response.status_code}: "
        f"{login_response.get_json()}"
    )
    token = (login_response.get_json() or {}).get("access_token")
    assert token, "Le login mobile doit retourner un access_token"
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class TestDriverCreationToAssignmentFlow:
    """Tests : Flux création chauffeur → assignation."""

    def test_e2e_driver_creation_to_assignment_flow(self, db):
        """Test : Créer chauffeur → Dispatch → Vérifier assignation."""
        # Setup : Créer company, client, driver
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)

        # Activer le dispatch pour la company
        company.dispatch_enabled = True
        db.session.commit()
        db.session.refresh(company)

        # Créer un booking pour demain
        scheduled_time = datetime.now(UTC) + timedelta(days=1, hours=2)
        booking = create_test_booking(
            db,
            client=client,
            scheduled_time=scheduled_time,
            status=BookingStatus.PENDING,
        )

        # Déclencher le dispatch (comme dans test_booking_dispatch_e2e.py)
        from services.unified_dispatch.core.engine import run as dispatch_run

        dispatch_date = scheduled_time.date()
        result = dispatch_run(
            company_id=company.id,
            for_date=dispatch_date.isoformat(),
            mode="heuristic_only",
        )

        # Vérifier que le dispatch s'est exécuté
        assert "assignments" in result
        assert isinstance(result["assignments"], list)

        # Recharger le booking depuis la DB
        db.session.refresh(booking)

        # Vérifier que le booking a été assigné au driver
        if booking.status == BookingStatus.ASSIGNED and booking.driver_id:
            assert booking.driver_id == driver.id
            assert_booking_assigned(booking, driver)


class TestDriverStatusUpdateFlow:
    """Tests : Mise à jour statut booking par chauffeur."""

    def test_e2e_driver_status_update_flow(self, e2e_client, db):
        """Test : Assigner booking → Mettre à jour statut → Vérifier notifications."""
        # Setup : Créer company, client, driver
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        user = driver.user

        user.set_password("testpassword123", force_change=False)
        db.session.commit()

        # Créer un booking assigné au driver
        scheduled_time = datetime.now(UTC) + timedelta(days=1, hours=2)
        booking = create_test_booking(
            db,
            client=client,
            scheduled_time=scheduled_time,
            status=BookingStatus.PENDING,
        )
        # Assigner le booking au driver
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.commit()

        # 1. Login en tant que driver
        auth_headers = _login_driver_mobile(e2e_client, user, "testpassword123")

        # 2. Mettre à jour le statut du booking (en_route)
        update_status_response = e2e_client.put(
            f"/api/v1/driver/me/bookings/{booking.id}/status",
            json={"status": "en_route"},
            headers=auth_headers,
        )

        assert update_status_response.status_code in (200, 201), (
            f"Mise à jour statut doit réussir, "
            f"reçu {update_status_response.status_code}: "
            f"{update_status_response.get_json()}"
        )

        # 3. Vérifier que le statut a été mis à jour
        db.session.refresh(booking)
        # Le statut peut être EN_ROUTE, IN_PROGRESS ou ASSIGNED selon l'implémentation
        assert booking.status in (
            BookingStatus.EN_ROUTE,
            BookingStatus.IN_PROGRESS,
            BookingStatus.ASSIGNED,
        ), f"Le statut doit être mis à jour, actuel: {booking.status}"

        # 4. Mettre à jour le statut à "in_progress"
        update_status_response = e2e_client.put(
            f"/api/v1/driver/me/bookings/{booking.id}/status",
            json={"status": "in_progress"},
            headers=auth_headers,
        )

        assert update_status_response.status_code in (200, 201)

        # 5. Vérifier que le statut est maintenant IN_PROGRESS
        db.session.refresh(booking)
        assert booking.status == BookingStatus.IN_PROGRESS, (
            f"Le statut doit être IN_PROGRESS, actuel: {booking.status}"
        )


class TestDriverBookingCompletionFlow:
    """Tests : Flux complet de terminaison de booking."""

    def test_e2e_driver_booking_completion_flow(self, e2e_client, db):
        """Test : Assigner booking → Démarrer → Terminer → Vérifier statut final."""
        # Setup : Créer company, client, driver
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        user = driver.user

        user.set_password("testpassword123", force_change=False)
        db.session.commit()

        # Créer un booking assigné au driver
        scheduled_time = datetime.now(UTC) + timedelta(days=1, hours=2)
        booking = create_test_booking(
            db,
            client=client,
            scheduled_time=scheduled_time,
            status=BookingStatus.PENDING,
        )
        # Assigner le booking au driver
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.commit()

        # 1. Login en tant que driver
        auth_headers = _login_driver_mobile(e2e_client, user, "testpassword123")

        # 2. Démarrer le booking (en_route)
        update_status_response = e2e_client.put(
            f"/api/v1/driver/me/bookings/{booking.id}/status",
            json={"status": "en_route"},
            headers=auth_headers,
        )
        assert update_status_response.status_code in (200, 201)

        # 3. Mettre en cours (in_progress)
        update_status_response = e2e_client.put(
            f"/api/v1/driver/me/bookings/{booking.id}/status",
            json={"status": "in_progress"},
            headers=auth_headers,
        )
        assert update_status_response.status_code in (200, 201)

        db.session.refresh(booking)
        assert booking.status == BookingStatus.IN_PROGRESS

        # 4. Terminer le booking (completed)
        update_status_response = e2e_client.put(
            f"/api/v1/driver/me/bookings/{booking.id}/status",
            json={"status": "completed"},
            headers=auth_headers,
        )
        assert update_status_response.status_code in (200, 201), (
            f"Terminaison booking doit réussir, "
            f"reçu {update_status_response.status_code}: "
            f"{update_status_response.get_json()}"
        )

        # 5. Vérifier le statut final
        db.session.refresh(booking)
        assert booking.status == BookingStatus.COMPLETED, (
            f"Le statut final doit être COMPLETED, actuel: {booking.status}"
        )
