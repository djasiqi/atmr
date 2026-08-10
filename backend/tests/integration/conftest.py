"""
Fixtures spécifiques pour les tests d'intégration DDD.

Ces fixtures étendent celles de backend/tests/conftest.py avec des données
spécifiques pour tester les flux complets route → use case → repository → DB.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import (
    Booking,
    Client,
    CompanyBillingSettings,
    Driver,
    Invoice,
    User,
)
from models.enums import (
    BookingStatus,
    ClientType,
    InvoiceStatus,
    ManagementMode,
    UserRole,
)


def get_db_dialect(db) -> str:
    """Retourne le nom du dialecte SQLAlchemy (postgresql, sqlite, etc.)."""
    try:
        bind = db.session.get_bind()
        return bind.dialect.name if bind else "unknown"
    except Exception:
        return "unknown"


@pytest.fixture
def requires_postgresql(db):
    """Skip si le dialecte n'est pas PostgreSQL. Pour tests d'intégration Postgres-only."""
    dialect = get_db_dialect(db)
    if dialect != "postgresql":
        msg = (
            f"PostgreSQL required for this integration test (current dialect: {dialect}). "
            + "Run: docker compose -f docker-compose.test.yml up -d postgres_test && "
            + "DATABASE_URL=postgresql://test:test@localhost:5433/atmr_test pytest ..."
        )
        pytest.skip(msg)


@pytest.fixture
def test_company(db, sample_company):
    """Entreprise autorisée par le jeton d'authentification d'intégration."""
    billing_settings = CompanyBillingSettings.query.filter_by(
        company_id=sample_company.id
    ).first()
    if billing_settings is None:
        billing_settings = CompanyBillingSettings(
            company_id=sample_company.id,
            payment_terms_days=30,
            invoice_prefix="INV",
            overdue_fee=Decimal("10.00"),
            reminder1_fee=Decimal("5.00"),
            reminder2_fee=Decimal("10.00"),
            reminder3_fee=Decimal("15.00"),
        )
        db.session.add(billing_settings)
        db.session.flush()

    return sample_company


@pytest.fixture
def test_client(db, test_company):
    """Client de test associé à l'entreprise."""
    if not test_company:
        pytest.skip("test_company required")

    unique_suffix = str(uuid.uuid4())[:8]

    # Créer un User d'abord (requis pour Client)
    user = User()
    user.public_id = str(uuid.uuid4())
    user.username = f"client_{unique_suffix}"
    user.email = f"client_{unique_suffix}@test.ch"
    user.role = UserRole.CLIENT
    user.first_name = f"Test{unique_suffix}"
    user.last_name = "Client"
    from ext import bcrypt

    user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
    db.session.add(user)
    db.session.flush()

    # Créer le Client avec relation user (préféré à user_id direct)
    client = Client()
    client.user = user  # Utiliser la relation plutôt que user_id directement
    client.company_id = test_company.id
    client.first_name = f"Test{unique_suffix}"
    client.last_name = "Client"
    client.email = f"client_{unique_suffix}@test.ch"
    # Les cas d'envoi de facture résolvent explicitement contact_email.
    client.contact_email = client.email
    client.phone = "0211234567"
    client.client_type = ClientType.TRANSPORT
    client.management_mode = ManagementMode.MANAGED
    db.session.add(client)
    db.session.flush()
    return client


@pytest.fixture
def test_driver(db, test_company):
    """Chauffeur de test associé à l'entreprise."""
    if not test_company:
        pytest.skip("test_company required")

    unique_suffix = str(uuid.uuid4())[:8]

    # Créer un User d'abord (requis pour Driver)
    user = User()
    user.username = f"driver_{unique_suffix}"
    user.email = f"driver_{unique_suffix}@test.ch"
    user.public_id = str(uuid.uuid4())
    user.role = UserRole.driver
    user.first_name = f"Test{unique_suffix}"
    user.last_name = "Driver"
    from ext import bcrypt

    user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
    db.session.add(user)
    db.session.flush()

    # Créer le Driver avec relation user (préféré à user_id direct)
    driver = Driver()
    driver.user = user  # Utiliser la relation plutôt que user_id directement
    driver.company_id = test_company.id
    driver.is_active = True
    db.session.add(driver)
    db.session.flush()
    return driver


@pytest.fixture
def test_booking(db, test_company, test_client):
    """Réservation de test complète."""
    if not test_company or not test_client:
        pytest.skip("test_company and test_client required")

    # S'assurer que test_client.user est chargé
    if not hasattr(test_client, "user") or test_client.user is None:
        db.session.refresh(test_client)
    assert test_client.user_id is not None, "test_client must have a user_id"

    booking = Booking()
    booking.user_id = test_client.user_id  # ✅ NOT NULL: utiliser user_id du client
    booking.company_id = test_company.id
    booking.client_id = test_client.id
    booking.customer_name = f"{test_client.first_name} {test_client.last_name}"
    booking.pickup_location = "Rue de Test 1, 1000 Lausanne"
    booking.dropoff_location = "Rue de Test 2, 1000 Lausanne"
    booking.scheduled_time = datetime.now(UTC) + timedelta(hours=1)
    booking.status = BookingStatus.PENDING
    booking.amount = Decimal("50.00")
    booking.vat_rate = Decimal("7.70")
    assert booking.user_id is not None, "booking.user_id must be set before flush"
    db.session.add(booking)
    db.session.flush()
    return booking


@pytest.fixture
def test_completed_booking(db, test_company, test_client):
    """Réservation complétée pour tests de facturation."""
    if not test_company or not test_client:
        pytest.skip("test_company and test_client required")

    # S'assurer que test_client.user est chargé
    if not hasattr(test_client, "user") or test_client.user is None:
        db.session.refresh(test_client)
    assert test_client.user_id is not None, "test_client must have a user_id"

    booking = Booking()
    booking.user_id = test_client.user_id  # ✅ NOT NULL: utiliser user_id du client
    booking.company_id = test_company.id
    booking.client_id = test_client.id
    booking.customer_name = f"{test_client.first_name} {test_client.last_name}"
    booking.pickup_location = "Rue de Test 1, 1000 Lausanne"
    booking.dropoff_location = "Rue de Test 2, 1000 Lausanne"
    booking.scheduled_time = datetime.now(UTC) - timedelta(days=1)
    booking.completed_at = datetime.now(UTC) - timedelta(hours=1)
    booking.status = BookingStatus.COMPLETED
    booking.amount = Decimal("100.00")
    booking.vat_rate = Decimal("7.70")
    booking.invoice_line_id = None  # Pas encore facturée
    assert booking.user_id is not None, "booking.user_id must be set before flush"
    db.session.add(booking)
    db.session.flush()
    return booking


@pytest.fixture
def test_invoice(db, test_company, test_client):
    """Facture de test."""
    if not test_company or not test_client:
        pytest.skip("test_company and test_client required")

    invoice = Invoice()
    invoice.company_id = test_company.id
    invoice.client_id = test_client.id
    invoice.invoice_number = f"INV-TEST-{uuid.uuid4().hex[:8]}"
    invoice.period_year = datetime.now(UTC).year
    invoice.period_month = datetime.now(UTC).month
    invoice.status = InvoiceStatus.DRAFT
    invoice.subtotal_amount = Decimal("100.00")
    invoice.vat_total_amount = Decimal("7.70")
    invoice.total_amount = Decimal("107.70")
    invoice.balance_due = Decimal("107.70")
    invoice.issued_at = datetime.now(UTC)
    invoice.due_date = datetime.now(UTC) + timedelta(days=30)
    db.session.add(invoice)
    db.session.flush()
    return invoice


@pytest.fixture
def authenticated_client(client, auth_headers):
    """Client Flask authentifié avec token JWT."""

    class AuthenticatedClient:  # noqa: D101
        """Wrapper pour client Flask avec authentification automatique.

        Note: Cette classe n'hérite d'aucune classe, donc pas besoin d'appeler
        super().__init__(). Le warning basedpyright est un faux positif.
        """

        def __init__(self, client, headers):  # noqa: D107  # pyright: ignore[reportMissingSuperCall]
            self._client = client
            self._headers = headers

        def get(self, url, **kwargs):
            """GET request avec authentification."""
            kwargs.setdefault("headers", {}).update(self._headers)
            return self._client.get(url, **kwargs)

        def post(self, url, **kwargs):
            """POST request avec authentification."""
            kwargs.setdefault("headers", {}).update(self._headers)
            return self._client.post(url, **kwargs)

        def put(self, url, **kwargs):
            """PUT request avec authentification."""
            kwargs.setdefault("headers", {}).update(self._headers)
            return self._client.put(url, **kwargs)

        def patch(self, url, **kwargs):
            """PATCH request avec authentification."""
            kwargs.setdefault("headers", {}).update(self._headers)
            return self._client.patch(url, **kwargs)

        def delete(self, url, **kwargs):
            """DELETE request avec authentification."""
            kwargs.setdefault("headers", {}).update(self._headers)
            return self._client.delete(url, **kwargs)

        def __getattr__(self, name):
            """Déléguer les autres méthodes au client original."""
            return getattr(self._client, name)

    return AuthenticatedClient(client, auth_headers)
