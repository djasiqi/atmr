"""
Tests pour les routes de réservations (bookings).
"""
from datetime import UTC, datetime, timedelta

import pytest

from models import Booking, BookingStatus, Client, User, UserRole


@pytest.fixture
def sample_client(db, sample_company):
    """Crée un client de test."""
    from ext import bcrypt
    user = User(
        username='clientuser',
        email='client@example.com',
        role=UserRole.client,
        first_name="Jean",
        last_name="Dupont",
        phone="0791234567",
        address="Rue Client 1, 1000 Lausanne"
    )
    user.password = bcrypt.generate_password_hash('password123').decode('utf-8')
    db.session.add(user)
    db.session.flush()

    client = Client(
        user_id=user.id,
        company_id=sample_company.id,
        billing_address="Rue Client 1, 1000 Lausanne",
        contact_email="client@example.com",
        contact_phone="0791234567"
    )
    db.session.add(client)
    db.session.commit()
    return client


def test_list_bookings_unauthenticated(client):
    """GET /bookings sans authentification renvoie 401."""
    response = client.get('/api/bookings/')
    assert response.status_code == 401


def test_list_bookings_authenticated(client, auth_headers, sample_user):
    """GET /bookings avec authentification renvoie liste de bookings."""
    response = client.get('/api/bookings/', headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert 'bookings' in data
    assert isinstance(data['bookings'], list)


def test_list_bookings_pagination(client, auth_headers, db, sample_user, sample_client):
    """GET /bookings?page=1&per_page=10 renvoie pagination."""
    # Créer quelques bookings de test
    for i in range(15):
        booking = Booking(
            client_id=sample_client.id,
            company_id=sample_user.company_id,
            user_id=sample_client.user_id,
            customer_name=f"Client {i}",
            pickup_location="Lausanne Gare",
            dropoff_location="CHUV",
            scheduled_time=datetime.now(UTC) + timedelta(days=i),
            status=BookingStatus.PENDING,
            amount=50.0,
            distance_meters=5000,
            duration_seconds=900
        )
        db.session.add(booking)
    db.session.commit()

    response = client.get('/api/bookings/?page=1&per_page=10', headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert 'bookings' in data
    assert len(data['bookings']) == 10  # Page 1 contient 10 éléments

    # Vérifier headers de pagination
    assert 'X-Total-Count' in response.headers
    assert 'X-Page' in response.headers


def test_get_booking_details(client, auth_headers, db, sample_user, sample_client):
    """GET /bookings/<id> renvoie les détails d'une réservation."""
    booking = Booking(
        client_id=sample_client.id,
        company_id=sample_user.company_id,
        user_id=sample_client.user_id,
        customer_name="Jean Dupont",
        pickup_location="Lausanne Gare",
        dropoff_location="CHUV",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=50.0,
        distance_meters=5000,
        duration_seconds=900
    )
    db.session.add(booking)
    db.session.commit()

    response = client.get(f'/api/bookings/{booking.id}', headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data['customer_name'] == "Jean Dupont"
    assert data['pickup_location'] == "Lausanne Gare"

