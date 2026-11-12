"""Helpers réutilisables pour les tests E2E.

Ce module fournit des fonctions utilitaires pour faciliter l'écriture
de tests end-to-end en encapsulant les opérations courantes :
- Authentification (login, logout, création de clients authentifiés)
- Création de données de test (company, client, driver, booking)
- Vérifications (assertions sur assignations, notifications, dispatch runs)
"""

import itertools
import secrets
from datetime import date
from typing import TYPE_CHECKING, Any

from flask import Flask
from flask.testing import FlaskClient
from flask_jwt_extended import create_access_token

from ext import db
from models import (
    Booking,
    BookingStatus,
    Client,
    Company,
    DispatchRun,
    DispatchStatus,
    Driver,
    User,
    UserRole,
)
from tests.conftest import persisted_fixture
from tests.factories import BookingFactory, ClientFactory, CompanyFactory, DriverFactory

if TYPE_CHECKING:
    from datetime import timedelta


# Départ aléatoire (isolation entre exécutions sur une DB non réinitialisée)
# puis incrément (unicité garantie à l'intérieur d'une session de tests).
_PHONE_SEQUENCE = itertools.count(secrets.randbelow(10**7))


def unique_phone(prefix: str = "+4179") -> str:
    """Génère un numéro suisse unique pour les inscriptions E2E.

    Les données créées via l'API (`/auth/register`) sont commitées et
    survivent aux tests : réutiliser un numéro en dur provoque un 409
    ``username_exists`` dès le second test qui l'emploie.

    Args:
        prefix: Préfixe international (défaut: mobile suisse ``+4179``)

    Returns:
        Numéro au format ``+41 7X XXX XX XX`` sans espaces

    Exemple:
        ```python
        register_payload = {"phone": unique_phone(), ...}
        ```
    """
    return f"{prefix}{next(_PHONE_SEQUENCE) % 10**7:07d}"


# =====================================================
# Helpers d'authentification
# =====================================================


def create_authenticated_client(
    app: Flask,
    user: User,
    expires_delta: "timedelta | None" = None,
) -> FlaskClient:
    """Crée un client Flask authentifié avec token JWT.

    Args:
        app: Instance Flask
        user: Utilisateur pour lequel créer le token
        expires_delta: Durée de validité du token (défaut: 24h)

    Returns:
        FlaskClient avec headers d'authentification configurés

    Exemple:
        ```python
        client = create_authenticated_client(app, sample_user)
        response = client.get("/api/v1/bookings")
        assert response.status_code == 200
        ```
    """
    from datetime import timedelta

    if expires_delta is None:
        expires_delta = timedelta(hours=24)

    claims = {
        "role": user.role.value,
        "company_id": getattr(user, "company_id", None),
        "driver_id": getattr(user, "driver_id", None),
        "aud": "atmr-api",
    }

    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
            expires_delta=expires_delta,
        )

    # Créer une classe wrapper qui ajoute automatiquement les headers
    # Cette classe n'hérite pas de FlaskClient, elle encapsule simplement le client
    # pour ajouter automatiquement les headers d'authentification
    class AuthenticatedClient:  # noqa: D101
        """Wrapper pour FlaskClient avec authentification automatique."""

        def __init__(self, client: FlaskClient, token: str) -> None:  # noqa: D107  # pyright: ignore[reportMissingSuperCall]
            # En Python 3, toutes les classes héritent implicitement d'object
            # et object.__init__() est un noop, donc pas besoin d'appeler
            # super().__init__()
            self._client = client
            self._token = token
            self._headers = {"Authorization": f"Bearer {token}"}

        def _add_headers(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            """Ajoute les headers d'authentification si non présents."""
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            kwargs["headers"].update(self._headers)
            return kwargs

        def get(self, *args: Any, **kwargs: Any) -> Any:
            kwargs = self._add_headers(kwargs)
            return self._client.get(*args, **kwargs)

        def post(self, *args: Any, **kwargs: Any) -> Any:
            kwargs = self._add_headers(kwargs)
            return self._client.post(*args, **kwargs)

        def put(self, *args: Any, **kwargs: Any) -> Any:
            kwargs = self._add_headers(kwargs)
            return self._client.put(*args, **kwargs)

        def patch(self, *args: Any, **kwargs: Any) -> Any:
            kwargs = self._add_headers(kwargs)
            return self._client.patch(*args, **kwargs)

        def delete(self, *args: Any, **kwargs: Any) -> Any:
            kwargs = self._add_headers(kwargs)
            return self._client.delete(*args, **kwargs)

    client = app.test_client()
    return AuthenticatedClient(client, token)  # type: ignore[return-value]


def login_as_user(client: FlaskClient, email: str, password: str) -> dict[str, Any]:
    """Connecte un utilisateur via l'API /auth/login.

    Args:
        client: Client Flask (non authentifié)
        email: Email de l'utilisateur
        password: Mot de passe

    Returns:
        Données de réponse du login (token, user, etc.)

    Raises:
        AssertionError: Si le login échoue

    Exemple:
        ```python
        response_data = login_as_user(client, "user@example.com", "password123")
        token = response_data["token"]
        ```
    """
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, (
        f"Login failed: {response.status_code} - {response.get_json()}"
    )
    data = response.get_json()
    assert data is not None
    return data


def logout_user(client: FlaskClient) -> None:
    """Déconnecte un utilisateur via l'API /auth/logout.

    Args:
        client: Client Flask authentifié

    Exemple:
        ```python
        logout_user(authenticated_client)
        # Les requêtes suivantes devraient retourner 401
        ```
    """
    response = client.post("/api/v1/auth/logout")
    assert response.status_code in (200, 204), f"Logout failed: {response.status_code}"


# =====================================================
# Helpers de création de données
# =====================================================


def create_test_company(db_session: Any) -> Company:
    """Crée une company de test persistée en DB.

    Args:
        db_session: Session DB (fixture pytest)

    Returns:
        Company créée et commitée

    Exemple:
        ```python
        company = create_test_company(db)
        assert company.id is not None
        ```
    """
    company = CompanyFactory()
    # CompanyFactory lie un User ADMIN par défaut ; les routes /companies/me/*
    # exigent le rôle company — aligner le rôle pour les fixtures E2E.
    if company.user is not None:
        company.user.role = UserRole.company
        db_session.session.flush()
    return persisted_fixture(db_session, company, Company)


def create_test_client(
    db_session: Any,
    company: Company | None = None,
) -> Client:
    """Crée un client de test persisté en DB.

    Args:
        db_session: Session DB (fixture pytest)
        company: Company à laquelle associer le client (créée si None)

    Returns:
        Client créé et commité

    Exemple:
        ```python
        client = create_test_client(db, company)
        assert client.company_id == company.id
        ```
    """
    if company is None:
        company = create_test_company(db_session)

    client = ClientFactory(company=company)
    return persisted_fixture(db_session, client, Client)


def create_test_driver(
    db_session: Any,
    company: Company | None = None,
) -> Driver:
    """Crée un chauffeur de test persisté en DB.

    Args:
        db_session: Session DB (fixture pytest)
        company: Company à laquelle associer le chauffeur (créée si None)

    Returns:
        Driver créé et commité

    Exemple:
        ```python
        driver = create_test_driver(db, company)
        assert driver.company_id == company.id
        ```
    """
    if company is None:
        company = create_test_company(db_session)

    driver = DriverFactory(company=company)
    return persisted_fixture(db_session, driver, Driver)


def create_test_booking(
    db_session: Any,
    client: Client | None = None,
    **kwargs: Any,
) -> Booking:
    """Crée un booking de test persisté en DB.

    Args:
        db_session: Session DB (fixture pytest)
        client: Client propriétaire du booking (créé si None)
        **kwargs: Arguments supplémentaires pour BookingFactory
                  (ex: scheduled_time, pickup_location, etc.)

    Returns:
        Booking créé et commité

    Exemple:
        ```python
        booking = create_test_booking(
            db,
            client,
            scheduled_time=datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
            pickup_location="Genève",
        )
        ```
    """
    if client is None:
        company = create_test_company(db_session)
        client = create_test_client(db_session, company=company)

    booking = BookingFactory(client=client, company=client.company, **kwargs)
    return persisted_fixture(db_session, booking, Booking)


# =====================================================
# Helpers de vérification
# =====================================================


def assert_booking_assigned(booking: Booking, driver: Driver) -> None:
    """Vérifie qu'un booking est assigné à un chauffeur.

    Args:
        booking: Booking à vérifier
        driver: Chauffeur attendu

    Raises:
        AssertionError: Si le booking n'est pas assigné au driver

    Exemple:
        ```python
        assert_booking_assigned(booking, driver)
        ```
    """
    # Recharger depuis DB pour s'assurer d'avoir les données à jour
    db.session.refresh(booking)
    assert booking.driver_id == driver.id, (
        f"Booking {booking.id} should be assigned to driver {driver.id}, "
        f"but driver_id is {booking.driver_id}"
    )
    assert booking.status == BookingStatus.ASSIGNED, (
        f"Booking {booking.id} should have status ASSIGNED, "
        f"but status is {booking.status}"
    )


def assert_notification_sent(user_id: int, event_type: str) -> None:
    """Vérifie qu'une notification a été envoyée à un utilisateur.

    ⚠️ NOTE: Cette fonction nécessite un mock de Socket.IO ou une table
    de notifications en DB. Pour l'instant, elle est un placeholder
    qui peut être étendue selon l'implémentation.

    Args:
        user_id: ID de l'utilisateur qui devrait avoir reçu la notification
        event_type: Type d'événement attendu (ex: "booking:assigned")

    Raises:
        AssertionError: Si la notification n'a pas été envoyée
        NotImplementedError: Si la vérification n'est pas implémentée

    Exemple:
        ```python
        # Avec mock de socketio
        with patch("services.realtime.socketio.socketio") as mock_socketio:
            # ... faire une action qui déclenche une notification ...
            assert_notification_sent(user_id, "booking:assigned")
            # Vérifier que mock_socketio.emit a été appelé
        ```
    """
    _ = user_id  # Utilisé dans la docstring
    _ = event_type  # Utilisé dans la docstring
    # TODO: Implémenter selon l'architecture de notifications
    # Option 1: Mock Socket.IO et vérifier les appels
    # Option 2: Table de notifications en DB
    # Option 3: Logs structurés
    raise NotImplementedError(
        "assert_notification_sent is a placeholder. "
        + "Implement according to your notification architecture "
        + "(mock Socket.IO, DB table, or structured logs)."
    )


def assert_dispatch_run_created(
    company_id: int,
    for_date: date,
    expected_status: DispatchStatus | None = None,
) -> DispatchRun:
    """Vérifie qu'un DispatchRun a été créé pour une company et une date.

    Args:
        company_id: ID de la company
        for_date: Date pour laquelle le dispatch run devrait exister
        expected_status: Statut attendu (None pour accepter n'importe quel statut)

    Returns:
        DispatchRun trouvé

    Raises:
        AssertionError: Si aucun DispatchRun n'est trouvé ou si le statut ne
            correspond pas

    Exemple:
        ```python
        dispatch_run = assert_dispatch_run_created(company.id, date(2025, 1, 15))
        assert dispatch_run.status == DispatchStatus.COMPLETED
        ```
    """
    dispatch_run = (
        db.session.query(DispatchRun)
        .filter_by(company_id=company_id, day=for_date)
        .first()
    )
    assert dispatch_run is not None, (
        f"No DispatchRun found for company_id={company_id} and day={for_date}"
    )

    if expected_status is not None:
        assert dispatch_run.status == expected_status, (
            f"DispatchRun {dispatch_run.id} should have status {expected_status}, "
            f"but status is {dispatch_run.status}"
        )

    return dispatch_run
