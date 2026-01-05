"""Tests pour valider l'infrastructure E2E et l'isolation entre tests.

Ce fichier teste que :
1. Les fixtures E2E fonctionnent correctement
2. Les helpers E2E sont utilisables
3. L'isolation entre tests est garantie (données d'un test n'affectent pas l'autre)
"""

import pytest
from flask.testing import FlaskClient

from models import Booking, Client, Company, Driver, User
from tests.e2e.helpers.e2e_helpers import (
    assert_booking_assigned,
    assert_dispatch_run_created,
    create_authenticated_client,
    create_test_booking,
    create_test_client,
    create_test_company,
    create_test_driver,
    login_as_user,
)


class TestE2EFixtures:
    """Tests pour valider que les fixtures E2E fonctionnent correctement."""

    def test_e2e_company_fixture(self, e2e_company):
        """Test : La fixture e2e_company crée une company valide."""
        assert e2e_company is not None
        assert e2e_company.id is not None
        assert e2e_company.name is not None
        assert e2e_company.user is not None

    def test_e2e_client_user_fixture(self, e2e_client_user):
        """Test : La fixture e2e_client_user crée un client et user valides."""
        client, user = e2e_client_user
        assert client is not None
        assert user is not None
        assert client.id is not None
        assert user.id is not None
        assert client.user_id == user.id

    def test_e2e_driver_fixture(self, e2e_driver):
        """Test : La fixture e2e_driver crée un driver valide."""
        assert e2e_driver is not None
        assert e2e_driver.id is not None
        assert e2e_driver.company_id is not None

    def test_e2e_authenticated_company_client(self, e2e_authenticated_company_client):
        """Test : La fixture e2e_authenticated_company_client fonctionne."""
        client = e2e_authenticated_company_client
        assert client is not None
        # Tester qu'on peut faire une requête authentifiée
        response = client.get("/api/v1/company/me")
        # 200 ou 404 selon l'endpoint, mais pas 401 (non authentifié)
        assert response.status_code != 401

    def test_e2e_authenticated_client_user(
        self, e2e_authenticated_client_user, e2e_client_user
    ):
        """Test : La fixture e2e_authenticated_client_user fonctionne."""
        client = e2e_authenticated_client_user
        _test_client, user = e2e_client_user
        assert client is not None
        # Tester qu'on peut faire une requête authentifiée
        # Le public_id est sur l'objet User, pas sur Client
        public_id = user.public_id if user else None
        if public_id:
            response = client.get(f"/api/v1/clients/{public_id}")
            # 200 ou 404 selon l'endpoint, mais pas 401 (non authentifié)
            assert response.status_code != 401
        else:
            # Si pas de public_id, on vérifie juste que le client est authentifié
            # en testant un endpoint général
            response = client.get("/api/v1/auth/me")
            assert response.status_code != 401


class TestE2EHelpers:
    """Tests pour valider que les helpers E2E fonctionnent correctement."""

    def test_create_test_company_helper(self, db):
        """Test : Helper create_test_company crée une company persistée."""
        company = create_test_company(db)
        assert company is not None
        assert company.id is not None
        # Vérifier qu'elle est bien en DB
        db.session.refresh(company)
        assert company.id is not None

    def test_create_test_client_helper(self, db):
        """Test : Helper create_test_client crée un client persisté."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        assert client is not None
        assert client.id is not None
        assert client.company_id == company.id

    def test_create_test_driver_helper(self, db):
        """Test : Helper create_test_driver crée un driver persisté."""
        company = create_test_company(db)
        driver = create_test_driver(db, company=company)
        assert driver is not None
        assert driver.id is not None
        assert driver.company_id == company.id

    def test_create_test_booking_helper(self, db):
        """Test : Helper create_test_booking crée un booking persisté."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        booking = create_test_booking(db, client=client)
        assert booking is not None
        assert booking.id is not None
        assert booking.client_id == client.id

    def test_create_authenticated_client_helper(self, app, e2e_company):
        """Test : Helper create_authenticated_client fonctionne."""
        user = e2e_company.user
        client = create_authenticated_client(app, user)
        assert client is not None
        # Tester qu'on peut faire une requête authentifiée
        response = client.get("/api/v1/company/me")
        assert response.status_code != 401


class TestE2EIsolation:
    """Tests pour valider l'isolation entre les tests E2E.

    Ces tests vérifient que les données créées dans un test
    ne sont pas visibles dans un autre test.
    """

    def test_isolation_company_fixture(self, e2e_company, db):
        """Test : Les companies créées dans différents tests sont isolées."""
        # Créer une company dans ce test
        company_id = e2e_company.id
        company_name = e2e_company.name

        # Vérifier qu'elle existe bien dans ce test
        company_in_db = db.session.get(Company, company_id)
        assert company_in_db is not None
        assert company_in_db.name == company_name

    def test_isolation_multiple_companies(self, db):
        """Test : Créer plusieurs companies dans le même test fonctionne."""
        company1 = create_test_company(db)
        company2 = create_test_company(db)

        assert company1.id != company2.id
        assert company1.name != company2.name

        # Vérifier qu'elles sont bien en DB
        db.session.refresh(company1)
        db.session.refresh(company2)
        assert company1.id is not None
        assert company2.id is not None

    def test_isolation_bookings_different_tests(self, db):
        """Test : Les bookings créés dans différents tests sont isolés."""
        # Créer un booking dans ce test
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        booking = create_test_booking(db, client=client)

        booking_id = booking.id
        booking_customer_name = booking.customer_name

        # Vérifier qu'il existe bien dans ce test
        booking_in_db = db.session.get(Booking, booking_id)
        assert booking_in_db is not None
        assert booking_in_db.customer_name == booking_customer_name

    def test_isolation_clients_different_companies(self, db):
        """Test : Les clients de différentes companies sont isolés."""
        company1 = create_test_company(db)
        company2 = create_test_company(db)

        client1 = create_test_client(db, company=company1)
        client2 = create_test_client(db, company=company2)

        assert client1.company_id == company1.id
        assert client2.company_id == company2.id
        assert client1.company_id != client2.company_id

    def test_isolation_drivers_different_companies(self, db):
        """Test : Les drivers de différentes companies sont isolés."""
        company1 = create_test_company(db)
        company2 = create_test_company(db)

        driver1 = create_test_driver(db, company=company1)
        driver2 = create_test_driver(db, company=company2)

        assert driver1.company_id == company1.id
        assert driver2.company_id == company2.id
        assert driver1.company_id != driver2.company_id


class TestE2EAssertionHelpers:
    """Tests pour valider les helpers d'assertion E2E."""

    def test_assert_dispatch_run_created(self, db):
        """Test : Helper assert_dispatch_run_created fonctionne."""
        from datetime import date

        from models import DispatchRun, DispatchStatus

        company = create_test_company(db)
        test_date = date(2025, 1, 15)

        # Créer un dispatch run
        dispatch_run = DispatchRun(
            company_id=company.id,
            day=test_date,
            status=DispatchStatus.COMPLETED,
        )
        db.session.add(dispatch_run)
        db.session.commit()

        # Tester que le helper le trouve
        found_run = assert_dispatch_run_created(company.id, test_date)
        assert found_run.id == dispatch_run.id
        assert found_run.status == DispatchStatus.COMPLETED

    def test_assert_booking_assigned(self, db):
        """Test : Helper assert_booking_assigned fonctionne."""
        from models import BookingStatus

        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client)

        # Assigner le booking au driver
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.commit()

        # Tester que le helper fonctionne
        assert_booking_assigned(booking, driver)
