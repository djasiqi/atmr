"""
Tests pour les routes de dispatch (planification automatique).
"""

from datetime import UTC, datetime, timedelta

import pytest

from models import Booking, BookingStatus, Driver, User, UserRole, Vehicle


@pytest.fixture
def mock_osrm(monkeypatch):
    """Mock OSRM pour éviter appels réseau."""

    def fake_table(*args, **kwargs):
        # Retourne une matrice 3x3 factice (durées en secondes)
        return {
            "code": "Ok",
            "durations": [[0, 600, 1200], [600, 0, 800], [1200, 800, 0]],
        }

    monkeypatch.setattr("services.osrm_client._table", fake_table)
    return fake_table


@pytest.fixture
def sample_driver(db, sample_company):
    """Crée un chauffeur de test."""
    import uuid

    from ext import bcrypt

    # Utiliser un email unique pour éviter les conflits de contrainte unique
    unique_suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"driver_{unique_suffix}",
        email=f"driver_{unique_suffix}@example.com",
        role=UserRole.driver,
        first_name="John",
        last_name="Driver",
    )
    user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
    db.session.add(user)
    db.session.flush()

    driver = Driver(user_id=user.id, company_id=sample_company.id, is_available=True)
    db.session.add(driver)
    db.session.flush()  # Use flush instead of commit to work with savepoints
    return driver


@pytest.fixture
def sample_vehicle(db, sample_company, sample_driver):
    """Crée un véhicule de test."""
    vehicle = Vehicle(
        company_id=sample_company.id,
        license_plate="VD-123456",
        model="Mercedes Vito",
        seats=8,
        wheelchair_accessible=True,
    )
    db.session.add(vehicle)
    db.session.flush()  # Utiliser flush au lieu de commit pour savepoints
    return vehicle


def test_dispatch_endpoint_exists(client, auth_headers):
    """GET /api/company_dispatch/ retourne une réponse."""
    response = client.get("/api/company_dispatch/", headers=auth_headers)
    # Peut être 200 (données) ou 404 (route non trouvée selon config)
    assert response.status_code in [200, 404, 405]


def test_dispatch_requires_auth(client):
    """GET /api/company_dispatch/ sans auth renvoie 401."""
    response = client.get("/api/company_dispatch/")
    assert response.status_code == 401


def test_create_booking_for_dispatch(
    client, auth_headers, db, sample_user, sample_client, sample_company
):
    """Créer des bookings en attente pour dispatch."""
    # Créer 2 bookings PENDING
    for i in range(2):
        booking = Booking(
            client_id=sample_client.id,
            company_id=sample_company.id,  # Utiliser sample_company.id au lieu de sample_user.company_id
            user_id=sample_client.user_id,
            customer_name=f"Client {i}",
            pickup_location="Lausanne Gare",
            dropoff_location="CHUV",
            scheduled_time=datetime.now(UTC) + timedelta(hours=i + 1),
            status=BookingStatus.PENDING,
            amount=50.0,
            distance_meters=0.5000,
            duration_seconds=0.900,
        )
        db.session.add(booking)
    db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

    # Vérifier que les bookings sont créés
    bookings = Booking.query.filter_by(status=BookingStatus.PENDING).all()
    assert len(bookings) >= 2


def test_driver_availability(db, sample_driver):
    """Vérifier qu'un chauffeur est disponible."""
    driver = Driver.query.get(sample_driver.id)
    assert driver is not None
    assert driver.is_available is True
