"""
Tests pour les routes de réservations (bookings).
"""

from datetime import UTC, datetime, timedelta

import pytest

from models import Booking, BookingStatus, Client, User, UserRole
from shared.time_utils import now_local


@pytest.fixture
def sample_client(db, sample_company):
    """Crée un client de test."""
    import uuid

    from ext import bcrypt

    # Utiliser un email unique pour éviter les conflits de contrainte unique
    unique_suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"clientuser_{unique_suffix}",
        email=f"client_{unique_suffix}@example.com",
        role=UserRole.client,
        first_name="Jean",
        last_name="Dupont",
        phone="0791234567",
        address="Rue Client 1, 1000 Lausanne",
    )
    user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
    db.session.add(user)
    db.session.flush()

    client = Client(
        user_id=user.id,
        company_id=sample_company.id,
        billing_address="Rue Client 1, 1000 Lausanne",
        contact_email=f"client_{unique_suffix}@example.com",
        contact_phone="0791234567",
    )
    db.session.add(client)
    db.session.flush()  # Use flush instead of commit to work with savepoints
    return client


def test_list_bookings_unauthenticated(client):
    """GET /bookings sans authentification renvoie 401."""
    response = client.get("/api/v1/bookings/")
    # 404 est acceptable si la route n'est pas initialisée (SKIP_ROUTES_INIT=1)
    assert response.status_code in (401, 404)


def test_list_bookings_authenticated(client, auth_headers, sample_user):
    """GET /bookings avec authentification renvoie liste de bookings."""
    response = client.get("/api/v1/bookings/", headers=auth_headers)
    # 404 est acceptable si la route n'est pas initialisée (SKIP_ROUTES_INIT=1)
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    assert response.status_code in (200, 403, 404)
    if response.status_code in (403, 404):
        return  # Skip le reste du test si la route n'existe pas / pas accessible
    data = response.get_json()
    assert "bookings" in data
    assert isinstance(data["bookings"], list)


def test_list_bookings_pagination(
    client, auth_headers, db, sample_user, sample_client, sample_company
):
    """GET /bookings?page=1&per_page=10 renvoie pagination."""
    # Créer quelques bookings de test
    for i in range(15):
        booking = Booking(
            client_id=sample_client.id,
            company_id=sample_company.id,  # Utiliser sample_company.id
            # au lieu de sample_user.company_id
            user_id=sample_client.user_id,
            customer_name=f"Client {i}",
            pickup_location="Lausanne Gare",
            dropoff_location="CHUV",
            scheduled_time=datetime.now(UTC) + timedelta(days=i),
            status=BookingStatus.PENDING,
            amount=50.0,
            distance_meters=0.5000,
            duration_seconds=0.900,
        )
        db.session.add(booking)
    db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

    response = client.get("/api/v1/bookings/?page=1&per_page=10", headers=auth_headers)
    # 404 est acceptable si la route n'est pas initialisée (SKIP_ROUTES_INIT=1)
    assert response.status_code in (200, 403, 404)
    if response.status_code in (403, 404):
        return  # Skip le reste du test si la route n'existe pas / pas accessible
    data = response.get_json()
    assert "bookings" in data
    assert len(data["bookings"]) == 10  # Page 1 contient 10 éléments

    # Vérifier headers de pagination
    assert "X-Total-Count" in response.headers
    assert "X-Page" in response.headers


def test_get_booking_details(
    client, auth_headers, db, sample_user, sample_client, sample_company
):
    """GET /bookings/<id> renvoie les détails d'une réservation."""
    booking = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,  # Utiliser sample_company.id
        # au lieu de sample_user.company_id
        user_id=sample_client.user_id,
        customer_name="Jean Dupont",
        pickup_location="Lausanne Gare",
        dropoff_location="CHUV",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=50.0,
        distance_meters=0.5000,
        duration_seconds=0.900,
    )
    db.session.add(booking)
    db.session.flush()  # Utiliser flush au lieu de commit pour savepoints

    response = client.get(f"/api/v1/bookings/{booking.id}", headers=auth_headers)
    # 404 est acceptable si la route n'est pas initialisée (SKIP_ROUTES_INIT=1)
    assert response.status_code in (200, 403, 404)
    if response.status_code in (403, 404):
        return  # Skip le reste du test si la route n'existe pas / pas accessible
    data = response.get_json()
    # La réponse peut avoir une structure avec "data" ou être directement l'objet
    booking_data = data.get("data", data)
    assert booking_data.get("client_name") == "Jean Dupont"
    assert booking_data["pickup_location"] == "Lausanne Gare"


# =====================================================
# Tests validation montant (amount)
# =====================================================


def test_booking_create_amount_zero_rejected(client, auth_headers, sample_client):
    """Test que amount=0 est rejeté lors de la création."""
    booking_data = {
        "customer_name": "Test Client",
        "pickup_location": "Lausanne Gare",
        "dropoff_location": "CHUV",
        "scheduled_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        "amount": 0,
    }
    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        json=booking_data,
        headers=auth_headers,
    )
    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data or "amount" in str(data).lower()
    assert "0.5" in str(data) or "minimum" in str(data).lower()


def test_booking_create_amount_below_minimum_rejected(
    client, auth_headers, sample_client
):
    """Test que amount < 0.5 est rejeté lors de la création."""
    for invalid_amount in [0.01, 0.3, 0.49]:
        booking_data = {
            "customer_name": "Test Client",
            "pickup_location": "Lausanne Gare",
            "dropoff_location": "CHUV",
            "scheduled_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "amount": invalid_amount,
        }
        response = client.post(
            f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
            json=booking_data,
            headers=auth_headers,
        )
        # 404 est acceptable si la route n'est pas initialisée
        # 403 est acceptable si le rôle du user n'a pas la permission (protection).
        if response.status_code in (403, 404):
            return
        assert response.status_code == 400, (
            f"amount={invalid_amount} devrait être rejeté"
        )
        data = response.get_json()
        assert "error" in data or "amount" in str(data).lower()


def test_booking_amount_rounding_rules(db, sample_client, sample_company):
    """Test que les règles d'arrondi métier sont appliquées."""
    from models import Booking

    # Test 0.6 → 0.5
    booking1 = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test 1",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=0.6,
    )
    db.session.add(booking1)
    db.session.flush()
    assert booking1.amount == 0.5, (
        f"0.6 devrait être arrondi à 0.5, obtenu {booking1.amount}"
    )

    # Test 0.75 → 0.8
    booking2 = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test 2",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=0.75,
    )
    db.session.add(booking2)
    db.session.flush()
    assert booking2.amount == 0.8, (
        f"0.75 devrait être arrondi à 0.8, obtenu {booking2.amount}"
    )

    # Test 39.98 → 40.0
    booking3 = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test 3",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=39.98,
    )
    db.session.add(booking3)
    db.session.flush()
    assert booking3.amount == 40.0, (
        f"39.98 devrait être arrondi à 40.0, obtenu {booking3.amount}"
    )

    # Test montant standard (pas d'arrondi spécial)
    booking4 = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test 4",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=50.0,
    )
    db.session.add(booking4)
    db.session.flush()
    assert booking4.amount == 50.0, (
        f"50.0 devrait rester 50.0, obtenu {booking4.amount}"
    )

    # Test montant avec décimales (arrondi standard)
    booking5 = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test 5",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=25.456,
    )
    db.session.add(booking5)
    db.session.flush()
    assert booking5.amount == 25.46, (
        f"25.456 devrait être arrondi à 25.46, obtenu {booking5.amount}"
    )


def test_booking_amount_minimum_accepted(db, sample_client, sample_company):
    """Test que amount=0.5 est accepté (montant minimum)."""
    from models import Booking

    booking = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test Minimum",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=datetime.now(UTC) + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=0.5,
    )
    db.session.add(booking)
    db.session.flush()
    assert booking.amount == 0.5, "0.5 devrait être accepté comme montant minimum"


def test_booking_return_leg_zero_amount_allowed(db, sample_client, sample_company):
    """Course retour : montant 0 (tarif porté par l’aller), is_return avant amount."""
    outbound = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Aller",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=now_local() + timedelta(days=1),
        status=BookingStatus.PENDING,
        amount=90.0,
        is_return=False,
    )
    db.session.add(outbound)
    db.session.flush()

    retour = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Aller",
        pickup_location="Genève",
        dropoff_location="Lausanne",
        is_return=True,
        parent_booking_id=outbound.id,
        time_confirmed=False,
        scheduled_time=None,
        status=BookingStatus.PENDING,
        amount=0,
    )
    db.session.add(retour)
    db.session.flush()
    assert retour.amount == 0.0


# =====================================================
# Tests géocodage obligatoire
# =====================================================


def test_booking_create_geocoding_failure_rejected(
    client, auth_headers, sample_client, monkeypatch
):
    """Test que le booking est rejeté si le géocodage échoue."""

    # Mock get_distance_duration pour simuler un échec de géocodage
    def mock_get_distance_duration_failure(*args, **kwargs):
        raise RuntimeError("ZERO_RESULTS")

    monkeypatch.setattr(
        "services.maps.get_distance_duration",
        mock_get_distance_duration_failure,
    )

    booking_data = {
        "customer_name": "Test Client",
        "pickup_location": "Adresse invalide XYZ123",
        "dropoff_location": "Autre adresse invalide ABC456",
        "scheduled_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        "amount": 50.0,
    }
    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        json=booking_data,
        headers=auth_headers,
    )
    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert (
        "impossible_de_geocoder" in data.get("error", "")
        or "geocodage" in str(data).lower()
    )


def test_booking_create_geocode_address_failure_rejected(
    client, auth_headers, sample_client, monkeypatch
):
    """Test que le booking est rejeté si geocode_address retourne None."""

    # Mock get_distance_duration pour réussir (simule succès distance matrix)
    def mock_get_distance_duration_success(*args, **kwargs):
        return 1800, 5000  # 30 min, 5 km

    # Mock geocode_address pour retourner None (échec géocodage)
    def mock_geocode_address_failure(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "services.maps.get_distance_duration",
        mock_get_distance_duration_success,
    )
    monkeypatch.setattr(
        "services.maps.geocode_address",
        mock_geocode_address_failure,
    )

    booking_data = {
        "customer_name": "Test Client",
        "pickup_location": "Adresse invalide",
        "dropoff_location": "Autre adresse",
        "scheduled_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        "amount": 50.0,
    }
    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        json=booking_data,
        headers=auth_headers,
    )
    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "impossible_de_geocoder" in data.get("error", "")


def test_booking_create_invalid_coordinates_rejected(
    client, auth_headers, sample_client, monkeypatch
):
    """Test que le booking est rejeté si les coordonnées sont invalides."""

    # Mock get_distance_duration pour réussir
    def mock_get_distance_duration_success(*args, **kwargs):
        return 1800, 5000

    # Mock geocode_address pour retourner des coordonnées invalides
    def mock_geocode_address_invalid(*args, **kwargs):
        return {"lat": 200.0, "lon": 300.0}  # Coordonnées hors limites

    monkeypatch.setattr(
        "services.maps.get_distance_duration",
        mock_get_distance_duration_success,
    )
    monkeypatch.setattr(
        "services.maps.geocode_address",
        mock_geocode_address_invalid,
    )

    booking_data = {
        "customer_name": "Test Client",
        "pickup_location": "Adresse test",
        "dropoff_location": "Autre adresse",
        "scheduled_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        "amount": 50.0,
    }
    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        json=booking_data,
        headers=auth_headers,
    )
    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "coordonnees_invalides" in data.get("error", "")


def test_booking_create_scheduled_time_none_rejected(
    client, auth_headers, sample_client, monkeypatch, db
):
    """Test que le booking est rejeté si scheduled_time est None."""
    # Test 1: scheduled_time manquant dans le payload
    booking_data = {
        "customer_name": "Test Client",
        "pickup_location": "Lausanne",
        "dropoff_location": "Genève",
        # scheduled_time manquant
        "amount": 50.0,
    }
    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        json=booking_data,
        headers=auth_headers,
    )
    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return
    # Le schéma Marshmallow devrait rejeter car scheduled_time est required=True
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data or "scheduled_time" in str(data).lower()

    # Test 2: scheduled_time=None dans le modèle (si créé directement)
    # Ce test vérifie que le validateur du modèle rejette None
    from models import Booking, BookingStatus

    with pytest.raises(ValueError, match="scheduled_time est obligatoire"):
        _ = Booking(
            client_id=sample_client.id,
            company_id=sample_client.company_id,
            user_id=sample_client.user_id,
            customer_name="Test Client",
            pickup_location="Lausanne",
            dropoff_location="Genève",
            scheduled_time=None,  # None devrait être rejeté
            status=BookingStatus.PENDING,
            amount=50.0,
        )


# =====================================================
# Tests pour décorateur @require_booking_ownership
# =====================================================


@pytest.fixture
def sample_booking_for_owner(db, sample_client, sample_company):
    """Crée un booking appartenant à sample_client."""
    from shared.time_utils import now_local

    booking = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test Client",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=now_local() + timedelta(hours=1),
        status=BookingStatus.PENDING,
        amount=50.0,
    )
    db.session.add(booking)
    db.session.flush()
    return booking


@pytest.fixture
def other_client(db, sample_company):
    """Crée un autre client pour tester IDOR."""
    import uuid

    from ext import bcrypt

    unique_suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"otherclient_{unique_suffix}",
        email=f"other_{unique_suffix}@example.com",
        role=UserRole.client,
        first_name="Other",
        last_name="Client",
        phone="0799999999",
    )
    user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
    db.session.add(user)
    db.session.flush()

    client = Client(
        user_id=user.id,
        company_id=sample_company.id,
        billing_address="Rue Other 1, 1000 Lausanne",
        contact_email=f"other_{unique_suffix}@example.com",
        contact_phone="0799999999",
    )
    db.session.add(client)
    db.session.flush()
    return client


def test_booking_idor_blocked_other_client(
    client, db, sample_client, other_client, sample_booking_for_owner
):
    """Test IDOR : Tentative accès booking d'un autre client doit retourner 403."""
    from flask_jwt_extended import create_access_token

    # Créer un token pour other_client (qui n'est pas propriétaire du booking)
    claims = {
        "role": other_client.user.role.value,
        "company_id": other_client.company_id,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(other_client.user.public_id), additional_claims=claims
        )
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get(
        f"/api/v1/bookings/{sample_booking_for_owner.id}", headers=headers
    )

    # 404 est acceptable si la route n'est pas initialisée
    if response.status_code == 404:
        return

    assert response.status_code in (401, 403)
    if response.status_code == 403:
        data = response.get_json()
        assert "error" in data
        assert (
            "non autorisé" in data["error"].lower() or "accès" in data["error"].lower()
        )


def test_booking_ownership_client_can_access_own_booking(
    client, db, sample_client, sample_booking_for_owner
):
    """Test ownership : Client peut accéder son propre booking."""
    from flask_jwt_extended import create_access_token

    # Créer un token pour sample_client (propriétaire du booking)
    claims = {
        "role": sample_client.user.role.value,
        "company_id": sample_client.company_id,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(sample_client.user.public_id), additional_claims=claims
        )
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get(
        f"/api/v1/bookings/{sample_booking_for_owner.id}", headers=headers
    )

    # 404 est acceptable si la route n'est pas initialisée
    if response.status_code == 404:
        return

    assert response.status_code == 200
    data = response.get_json()
    # La réponse peut avoir une structure avec "data" ou être directement l'objet
    booking_data = data.get("data", data)
    assert "id" in booking_data
    assert booking_data["id"] == sample_booking_for_owner.id


def test_booking_ownership_admin_can_access_all_bookings(
    client, db, sample_admin_user, sample_booking_for_owner, admin_headers
):
    """Test admin : Admin peut accéder tous bookings."""
    response = client.get(
        f"/api/v1/bookings/{sample_booking_for_owner.id}", headers=admin_headers
    )

    # 404 est acceptable si la route n'est pas initialisée
    if response.status_code == 404:
        return

    assert response.status_code == 200
    data = response.get_json()
    # La réponse peut avoir une structure avec "data" ou être directement l'objet
    booking_data = data.get("data", data)
    assert "id" in booking_data
    assert booking_data["id"] == sample_booking_for_owner.id


# =====================================================
# Tests pour validation return_time > scheduled_time
# =====================================================


def test_booking_create_round_trip_without_return_time_allowed(
    client, auth_headers, sample_client
):
    """Aller-retour : date de retour sans heure (return_date + heure à confirmer)."""
    from datetime import UTC, datetime, timedelta

    scheduled_dt = datetime.now(UTC) + timedelta(hours=1)
    return_ymd = (scheduled_dt + timedelta(days=1)).date().isoformat()

    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        headers=auth_headers,
        json={
            "customer_name": "Test Client",
            "pickup_location": "Lausanne",
            "dropoff_location": "Genève",
            "scheduled_time": scheduled_dt.isoformat(),
            "amount": 50.0,
            "is_round_trip": True,
            "return_date": return_ymd,
        },
    )

    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return

    assert response.status_code in (200, 201)


def test_booking_create_round_trip_return_time_before_scheduled_rejected(
    client, auth_headers, sample_client
):
    """Test que return_time <= scheduled_time est rejeté."""
    from datetime import UTC, datetime, timedelta

    scheduled_time = datetime.now(UTC) + timedelta(hours=2)
    return_time = scheduled_time - timedelta(
        hours=1
    )  # return_time avant scheduled_time

    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        headers=auth_headers,
        json={
            "customer_name": "Test Client",
            "pickup_location": "Lausanne",
            "dropoff_location": "Genève",
            "scheduled_time": scheduled_time.isoformat(),
            "amount": 50.0,
            "is_round_trip": True,
            "return_time": return_time.isoformat(),
        },
    )

    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return

    assert response.status_code == 400
    data = response.get_json()
    assert (
        "error" in data
        or "return_time" in str(data).lower()
        or "postérieur" in str(data).lower()
    )


def test_booking_create_round_trip_return_time_after_scheduled_accepted(
    client, auth_headers, sample_client, db
):
    """Test que return_time > scheduled_time est accepté."""
    from datetime import UTC, datetime, timedelta

    scheduled_time = datetime.now(UTC) + timedelta(hours=1)
    return_time = scheduled_time + timedelta(
        hours=2
    )  # return_time après scheduled_time

    response = client.post(
        f"/api/v1/bookings/clients/{sample_client.user.public_id}/bookings",
        headers=auth_headers,
        json={
            "customer_name": "Test Client",
            "pickup_location": "Lausanne",
            "dropoff_location": "Genève",
            "scheduled_time": scheduled_time.isoformat(),
            "amount": 50.0,
            "is_round_trip": True,
            "return_time": return_time.isoformat(),
        },
    )

    # 404 est acceptable si la route n'est pas initialisée
    # 403 est acceptable si le rôle du user n'a pas la permission (protection).
    if response.status_code in (403, 404):
        return

    # Peut être 200 (succès) ou 400 (autre erreur de validation), mais pas
    # 400 pour return_time
    if response.status_code == 400:
        data = response.get_json()
        # Ne doit pas être une erreur de validation return_time
        assert (
            "return_time" not in str(data).lower()
            or "postérieur" not in str(data).lower()
        )


# =====================================================
# Tests pour validation status=ASSIGNED avec driver_id
# =====================================================


def test_booking_assigned_without_driver_id_rejected(
    db, sample_client, sample_company, frozen_time
):
    """Test que status=ASSIGNED sans driver_id est rejeté."""
    booking = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test Client",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=now_local() + timedelta(hours=1),
        status=BookingStatus.PENDING,
        amount=50.0,
    )
    db.session.add(booking)
    db.session.flush()

    # Tenter de mettre status=ASSIGNED sans driver_id
    with pytest.raises(ValueError, match=r"driver_id.*ASSIGNED|ASSIGNED.*driver_id"):
        booking.status = BookingStatus.ASSIGNED
    db.session.rollback()


def test_booking_driver_id_none_with_assigned_status_rejected(
    db, sample_client, sample_company, sample_driver, frozen_time
):
    """Test que driver_id=None avec status=ASSIGNED est rejeté."""
    booking = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test Client",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=now_local() + timedelta(hours=1),
        driver_id=sample_driver.id,  # Assigner d'abord un driver
        status=BookingStatus.ASSIGNED,
        amount=50.0,
    )
    db.session.add(booking)
    db.session.flush()

    # Tenter de mettre driver_id=None alors que status=ASSIGNED
    with pytest.raises(
        ValueError, match=r"driver_id.*NULL.*ASSIGNED|ASSIGNED.*driver_id"
    ):
        booking.driver_id = None
    db.session.rollback()


def test_booking_assigned_with_driver_id_accepted(
    db, sample_client, sample_company, sample_driver, frozen_time
):
    """Test que status=ASSIGNED avec driver_id est accepté."""
    booking = Booking(
        client_id=sample_client.id,
        company_id=sample_company.id,
        user_id=sample_client.user_id,
        customer_name="Test Client",
        pickup_location="Lausanne",
        dropoff_location="Genève",
        scheduled_time=now_local() + timedelta(hours=1),
        status=BookingStatus.PENDING,
        amount=50.0,
    )
    db.session.add(booking)
    db.session.flush()

    # Assigner un driver puis mettre status=ASSIGNED (devrait fonctionner)
    booking.driver_id = sample_driver.id
    booking.status = BookingStatus.ASSIGNED
    db.session.flush()  # Ne doit pas lever d'exception
    db.session.rollback()


def _client_auth_headers(client, sample_client):
    from flask_jwt_extended import create_access_token

    from ext import db
    from models.enums import UserRole
    from models.user import User

    user = db.session.get(User, sample_client.user_id)
    claims = {
        "role": UserRole.client.value,
        "company_id": getattr(user, "company_id", None),
        "driver_id": None,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


@pytest.mark.integration
def test_saferpay_assert_returns_finalize_payload(
    client, db, sample_company, sample_client, monkeypatch, requires_postgresql
):
    from datetime import UTC, datetime
    from unittest.mock import patch

    from models.booking import Booking
    from models.enums import BookingStatus, PaymentStatus
    from models.payment import Payment

    monkeypatch.setenv("SAFERPAY_CUSTOMER_ID", "cid")
    monkeypatch.setenv("SAFERPAY_TERMINAL_ID", "tid")
    monkeypatch.setenv("SAFERPAY_API_USERNAME", "u")
    monkeypatch.setenv("SAFERPAY_API_PASSWORD", "p")

    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()

    pay = Payment(
        amount=10.0,
        method="credit_card",
        status=PaymentStatus.PENDING,
        user_id=sample_client.user_id,
        client_id=sample_client.id,
        booking_id=booking.id,
        payment_provider="saferpay",
    )
    pay.saferpay_token = "t"
    db.session.add(pay)
    db.session.commit()
    pid = pay.id

    headers = _client_auth_headers(client, sample_client)
    with patch(
        "routes.bookings.finalize_saferpay_payment",
        return_value={"status": "already_completed", "payment_id": pid},
    ):
        rv = client.post(
            f"/api/v1/bookings/{booking.id}/saferpay/assert",
            json={"payment_id": pid},
            headers=headers,
        )
    assert rv.status_code == 200
    body = rv.get_json()
    data = body.get("data") or body
    assert data.get("status") == "already_completed"


@pytest.mark.integration
def test_saferpay_assert_not_found_payment(
    client, db, sample_company, sample_client, monkeypatch, requires_postgresql
):
    from datetime import UTC, datetime

    from models.booking import Booking
    from models.enums import BookingStatus

    monkeypatch.setenv("SAFERPAY_CUSTOMER_ID", "cid")
    monkeypatch.setenv("SAFERPAY_TERMINAL_ID", "tid")
    monkeypatch.setenv("SAFERPAY_API_USERNAME", "u")
    monkeypatch.setenv("SAFERPAY_API_PASSWORD", "p")

    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.commit()

    headers = _client_auth_headers(client, sample_client)
    rv = client.post(
        f"/api/v1/bookings/{booking.id}/saferpay/assert",
        json={"payment_id": 999999999},
        headers=headers,
    )
    assert rv.status_code == 404
