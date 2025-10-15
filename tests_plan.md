# 🧪 Plan de Tests - ATMR (Backend + Frontend)

**Date**: 15 octobre 2025  
**Objectif**: Atteindre 70%+ coverage backend, 60%+ frontend, tests E2E critiques

---

## 📋 État Actuel (Estimé)

### Backend (Python/pytest)

**Coverage estimée**: <30%  
**Tests existants**:

- `backend/tests/test_dispatch_integration.py` (628 lignes) ✅
- `backend/tests/test_invoice_service.py` (présent mais incomplet)

**Gaps majeurs**:

- ❌ Routes auth (login, refresh, register, forgot-password)
- ❌ Routes bookings (CRUD, assign, cancel, round-trip)
- ❌ Routes companies (CRUD drivers, dispatch triggers)
- ❌ Services (invoice, PDF, QR-bill, OSRM, maps)
- ❌ Tasks Celery (billing, dispatch autorun)
- ❌ SocketIO handlers (chat, driver_location, rooms)
- ❌ Models validations (contraintes, validators)

### Frontend (React/Jest/RTL)

**Coverage estimée**: <20%  
**Tests existants**:

- `frontend/src/App.test.js` (test smoke basique)
- `frontend/src/setupTests.js` (config Jest)

**Gaps majeurs**:

- ❌ Pages (Dashboard Company/Driver/Client, Invoices, Dispatch, Planning)
- ❌ Composants (Modal, AddressAutocomplete, ChatWidget, DispatchDelayWidget)
- ❌ Hooks (useAuthToken, useCompanySocket, useDispatchStatus)
- ❌ Services (apiClient, companySocket, invoiceService)
- ❌ Utils (formatDate, validations, helpers)
- ❌ E2E (0 tests Cypress/Playwright)

---

## 🎯 Objectifs de Coverage

| Périmètre                  | Actuel | Cible             | Priorité |
| -------------------------- | ------ | ----------------- | -------- |
| **Backend - Routes**       | ~10%   | 80%               | Critique |
| **Backend - Services**     | ~20%   | 70%               | Critique |
| **Backend - Models**       | ~40%   | 85%               | Élevée   |
| **Backend - Tasks Celery** | 0%     | 60%               | Moyenne  |
| **Backend - SocketIO**     | 0%     | 50%               | Moyenne  |
| **Frontend - Pages**       | 0%     | 60%               | Élevée   |
| **Frontend - Components**  | 0%     | 70%               | Moyenne  |
| **Frontend - Hooks**       | 0%     | 80%               | Élevée   |
| **Frontend - Services**    | 0%     | 75%               | Critique |
| **E2E (Cypress)**          | 0%     | 5 tests critiques | Critique |

---

## 🧪 BACKEND: Plan de Tests Détaillé

### 1. Tests Routes (pytest + fixtures)

#### Fichier: `backend/tests/test_routes_auth.py`

**Périmètre**: Login, Refresh, Register, Forgot-password, Reset-password

```python
import pytest
from flask import Flask
from models import User, Client, UserRole, db
from ext import jwt

@pytest.fixture
def test_client(app: Flask):
    """Fixture client test Flask"""
    with app.test_client() as client:
        yield client

@pytest.fixture
def test_user(app: Flask):
    """Crée utilisateur test"""
    with app.app_context():
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.CLIENT
        )
        user.set_password("password123")
        db.session.add(user)

        client = Client(user_id=user.id, is_active=True)
        db.session.add(client)
        db.session.commit()

        yield user

        # Cleanup
        db.session.delete(client)
        db.session.delete(user)
        db.session.commit()

def test_login_success(test_client, test_user):
    """Test login avec credentials valides"""
    response = test_client.post('/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'token' in data
    assert 'refresh_token' in data
    assert data['user']['email'] == 'test@example.com'

def test_login_invalid_password(test_client, test_user):
    """Test login avec mauvais mot de passe"""
    response = test_client.post('/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'wrongpassword'
    })
    assert response.status_code == 401
    assert 'error' in response.get_json()

def test_login_rate_limit(test_client, test_user):
    """Test rate limiting (5 per minute)"""
    for _ in range(6):
        response = test_client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'wrongpassword'
        })
    assert response.status_code == 429  # Too Many Requests

def test_refresh_token(test_client, test_user, app):
    """Test refresh token génère nouveau access token"""
    with app.app_context():
        refresh_token = create_refresh_token(identity=test_user.public_id)

    response = test_client.post('/api/auth/refresh-token', headers={
        'Authorization': f'Bearer {refresh_token}'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'access_token' in data

def test_register_success(test_client):
    """Test inscription nouveau client"""
    response = test_client.post('/api/auth/register', json={
        'username': 'newuser',
        'email': 'newuser@example.com',
        'password': 'securepass123',
        'first_name': 'John',
        'last_name': 'Doe'
    })
    assert response.status_code == 201
    data = response.get_json()
    assert data['username'] == 'newuser'

    # Cleanup
    # (automatique via fixture app teardown)

def test_register_duplicate_email(test_client, test_user):
    """Test inscription email déjà existant"""
    response = test_client.post('/api/auth/register', json={
        'username': 'duplicate',
        'email': 'test@example.com',  # Déjà existant
        'password': 'pass123'
    })
    assert response.status_code == 409  # Conflict
```

**Cas de test total**: ~15-20 (login variations, register edge cases, forgot-password flow)

**Gain coverage**: Routes auth 0% → 85%

---

#### Fichier: `backend/tests/test_routes_bookings.py`

**Périmètre**: CRUD bookings, assign driver, cancel, round-trip

```python
import pytest
from datetime import datetime, timedelta
from models import Booking, BookingStatus, Client, Company, Driver, User, UserRole

@pytest.fixture
def company_user(app):
    """Crée entreprise test"""
    with app.app_context():
        user = User(username="company", email="company@test.com", role=UserRole.COMPANY)
        user.set_password("pass")
        db.session.add(user)
        db.session.flush()

        company = Company(user_id=user.id, name="Test Transport", is_approved=True)
        db.session.add(company)
        db.session.commit()

        yield company, user

        # Cleanup
        db.session.delete(company)
        db.session.delete(user)
        db.session.commit()

@pytest.fixture
def client_user(app, company_user):
    """Crée client test"""
    company, _ = company_user
    with app.app_context():
        user = User(username="client", email="client@test.com", role=UserRole.CLIENT)
        user.set_password("pass")
        db.session.add(user)
        db.session.flush()

        client = Client(user_id=user.id, company_id=company.id, is_active=True)
        db.session.add(client)
        db.session.commit()

        yield client, user

        # Cleanup...

def test_create_booking_success(test_client, client_user, mocker):
    """Test création booking avec géocodage"""
    client, user = client_user

    # Mock services.maps.get_distance_duration
    mocker.patch('services.maps.get_distance_duration', return_value=(1800, 15000))
    mocker.patch('services.maps.geocode_address', return_value=(46.2044, 6.1432))

    # Login
    login_resp = test_client.post('/api/auth/login', json={
        'email': user.email,
        'password': 'pass'
    })
    token = login_resp.get_json()['token']

    # Create booking
    response = test_client.post(f'/api/bookings/clients/{user.public_id}/bookings',
        json={
            'customer_name': 'Patient Test',
            'pickup_location': 'Rue de Genève 1, 1200 Genève',
            'dropoff_location': 'Hôpital Cantonal, 1211 Genève',
            'scheduled_time': (datetime.now() + timedelta(hours=2)).isoformat(),
            'amount': 45.50,
            'is_round_trip': False
        },
        headers={'Authorization': f'Bearer {token}'}
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data['pickup_location'] == 'Rue de Genève 1, 1200 Genève'
    assert data['status'] == 'PENDING'
    assert data['duration_seconds'] == 1800

def test_create_booking_round_trip(test_client, client_user, mocker):
    """Test création booking aller-retour"""
    # Similar setup + assert 2 bookings created
    pass

def test_assign_driver_to_booking(test_client, company_user, mocker):
    """Test assignation chauffeur à booking"""
    # Create booking + driver
    # POST /api/bookings/{id}/assign
    # Assert booking.status == ASSIGNED
    pass

def test_cancel_booking(test_client, client_user):
    """Test annulation booking"""
    # Create booking
    # DELETE /api/bookings/{id}
    # Assert booking.status == CANCELED
    pass

# ... 10-15 tests supplémentaires (validations, edge cases)
```

**Cas de test total**: ~20-25  
**Gain coverage**: Routes bookings 0% → 80%

---

### 2. Tests Services

#### Fichier: `backend/tests/test_service_invoice.py`

**Périmètre**: Génération factures, rappels, QR-bill, PDF

```python
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from services.invoice_service import InvoiceService
from models import Invoice, InvoiceLine, Company, Client, Booking

def test_generate_invoice_success(app, company_user, client_user, mocker):
    """Test génération facture période"""
    company, _ = company_user
    client, _ = client_user

    # Créer bookings de test
    with app.app_context():
        booking1 = Booking(
            customer_name="Patient",
            pickup_location="A",
            dropoff_location="B",
            scheduled_time=datetime(2025, 9, 15, 10, 0),
            amount=50.0,
            status=BookingStatus.COMPLETED,
            user_id=client.user_id,
            client_id=client.id,
            company_id=company.id
        )
        db.session.add(booking1)
        db.session.commit()

    # Mock PDF service
    mocker.patch('services.pdf_service.PDFService.generate_invoice_pdf',
                 return_value='http://localhost:5000/uploads/invoices/test.pdf')

    service = InvoiceService()
    invoice = service.generate_invoice(
        company_id=company.id,
        client_id=client.id,
        period_year=2025,
        period_month=9
    )

    assert invoice is not None
    assert invoice.subtotal_amount == Decimal('50.0')
    assert invoice.total_amount == Decimal('50.0')
    assert len(invoice.lines) == 1
    assert invoice.pdf_url is not None

def test_generate_invoice_no_bookings(app, company_user, client_user):
    """Test génération facture sans bookings → erreur"""
    service = InvoiceService()
    with pytest.raises(ValueError, match="Aucune réservation trouvée"):
        service.generate_invoice(
            company_id=company_user[0].id,
            client_id=client_user[0].id,
            period_year=2025,
            period_month=8  # Aucun booking ce mois
        )

def test_generate_reminder_level_1(app, mocker):
    """Test génération 1er rappel"""
    # Create overdue invoice
    # Call service.generate_reminder(invoice_id, level=1)
    # Assert reminder created, fee added
    pass

def test_qrbill_generation(app, mocker):
    """Test génération QR-bill avec IBAN"""
    # Mock qrbill library
    # Call qrbill_service.generate_qr_bill(invoice)
    # Assert QR reference valid, IBAN correct
    pass

# ... 10-15 tests supplémentaires
```

**Cas de test total**: ~18-22  
**Gain coverage**: Services invoice/PDF/QR-bill 20% → 75%

---

#### Fichier: `backend/tests/test_service_osrm.py`

**Périmètre**: Matrices, routes, fallback haversine, cache Redis

```python
import pytest
import responses  # Mock HTTP requests
from services.osrm_client import build_distance_matrix_osrm, eta_seconds

@responses.activate
def test_osrm_matrix_success():
    """Test matrice OSRM avec réponse valide"""
    coords = [(46.2044, 6.1432), (46.5197, 6.6323)]  # Genève, Lausanne

    # Mock OSRM HTTP response
    responses.add(
        responses.GET,
        'http://localhost:5000/table/v1/driving/6.1432,46.2044;6.6323,46.5197',
        json={
            'code': 'Ok',
            'durations': [[0, 2400], [2400, 0]]
        },
        status=200
    )

    matrix = build_distance_matrix_osrm(
        coords,
        base_url='http://localhost:5000',
        redis_client=None  # Pas de cache pour test
    )

    assert matrix[0][1] == 2400.0  # 40 min
    assert matrix[1][0] == 2400.0
    assert matrix[0][0] == 0.0

@responses.activate
def test_osrm_matrix_fallback_haversine():
    """Test fallback haversine si OSRM down"""
    coords = [(46.2044, 6.1432), (46.5197, 6.6323)]

    # Mock OSRM error
    responses.add(
        responses.GET,
        'http://localhost:5000/table/v1/driving/6.1432,46.2044;6.6323,46.5197',
        json={'error': 'Service unavailable'},
        status=503
    )

    matrix = build_distance_matrix_osrm(
        coords,
        base_url='http://localhost:5000',
        redis_client=None,
        max_retries=0  # Pas de retry pour test rapide
    )

    # Haversine: ~60km → ~2400s à 25km/h (avg_kmh fallback)
    assert 2000 < matrix[0][1] < 3000

def test_osrm_cache_hit(mocker, redis_mock):
    """Test cache Redis hit (pas d'appel HTTP)"""
    coords = [(46.2044, 6.1432), (46.5197, 6.6323)]

    # Mock Redis GET retourne résultat cached
    redis_mock.get.return_value = '{"durations": [[0, 2400], [2400, 0]]}'

    matrix = build_distance_matrix_osrm(
        coords,
        base_url='http://localhost:5000',
        redis_client=redis_mock
    )

    assert matrix[0][1] == 2400.0
    redis_mock.get.assert_called_once()  # Cache hit

# ... 8-12 tests supplémentaires (retry, timeout, singleflight)
```

**Cas de test total**: ~12-15  
**Gain coverage**: OSRM client 0% → 80%

---

### 3. Tests Models (Validations & Contraintes)

#### Fichier: `backend/tests/test_models_booking.py`

```python
import pytest
from datetime import datetime, timedelta
from models import Booking, BookingStatus
from shared.time_utils import now_local

def test_booking_validates_user_id(app):
    """Test validation user_id positif"""
    with app.app_context():
        with pytest.raises(ValueError, match="ID utilisateur doit être un entier positif"):
            booking = Booking(
                customer_name="Test",
                pickup_location="A",
                dropoff_location="B",
                scheduled_time=now_local() + timedelta(hours=1),
                amount=50,
                user_id=-1,  # ❌ Invalide
                client_id=1
            )

def test_booking_validates_scheduled_time_future(app):
    """Test validation scheduled_time futur"""
    with app.app_context():
        with pytest.raises(ValueError, match="Heure prévue dans le passé"):
            booking = Booking(
                customer_name="Test",
                pickup_location="A",
                dropoff_location="B",
                scheduled_time=now_local() - timedelta(hours=1),  # ❌ Passé
                amount=50,
                user_id=1,
                client_id=1
            )

def test_booking_enforce_billing_exclusive(app):
    """Test hook billed_to_company_id obligatoire si type != patient"""
    with app.app_context():
        with pytest.raises(ValueError, match="billed_to_company_id est obligatoire"):
            booking = Booking(
                customer_name="Test",
                pickup_location="A",
                dropoff_location="B",
                scheduled_time=now_local() + timedelta(hours=1),
                amount=50,
                user_id=1,
                client_id=1,
                billed_to_type="clinic",  # Type != patient
                billed_to_company_id=None  # ❌ Manquant
            )
            db.session.add(booking)
            db.session.commit()  # Trigger hook before_insert

# ... 10-15 tests validations supplémentaires
```

**Cas de test total**: ~15-20 par model (User, Company, Invoice, etc.)  
**Gain coverage**: Models 40% → 85%

---

### 4. Tests Tasks Celery

#### Fichier: `backend/tests/test_tasks_dispatch.py`

```python
import pytest
from datetime import date
from tasks.dispatch_tasks import run_dispatch_task, autorun_tick
from models import Company, DispatchRun

def test_run_dispatch_task_success(app, company_user, mocker):
    """Test tâche dispatch complète"""
    company, _ = company_user

    # Mock engine.run
    mocker.patch('services.unified_dispatch.engine.run', return_value={
        'assignments': [{'booking_id': 1, 'driver_id': 1}],
        'unassigned': [],
        'meta': {'reason': 'success'},
        'dispatch_run_id': 1
    })

    result = run_dispatch_task(
        company_id=company.id,
        for_date=str(date.today()),
        mode='auto'
    )

    assert 'assignments' in result
    assert len(result['assignments']) == 1
    assert result['meta']['reason'] == 'success'

def test_run_dispatch_task_retry_on_db_error(app, mocker):
    """Test retry sur erreur DB transitoire"""
    from sqlalchemy.exc import OperationalError

    mock_engine = mocker.patch('services.unified_dispatch.engine.run')
    mock_engine.side_effect = OperationalError("Connection lost", None, None)

    with pytest.raises(Exception):  # Retry puis MaxRetriesExceeded
        run_dispatch_task(company_id=1, for_date=str(date.today()))

    assert mock_engine.call_count > 1  # Retry tenté

def test_autorun_tick(app, mocker):
    """Test tick autorun (toutes entreprises dispatch_enabled)"""
    # Create 2 companies (1 enabled, 1 disabled)
    # Mock run_dispatch_task.delay
    # Assert 1 seul dispatch déclenché
    pass

# ... 8-10 tests supplémentaires
```

**Cas de test total**: ~10-12  
**Gain coverage**: Tasks Celery 0% → 65%

---

### 5. Tests SocketIO

#### Fichier: `backend/tests/test_socketio_chat.py`

```python
import pytest
from flask_socketio import SocketIOTestClient

def test_socketio_connect_driver(app, socketio, driver_user):
    """Test connexion driver avec JWT"""
    driver, user = driver_user
    token = create_access_token(identity=user.public_id)

    client = socketio.test_client(app, auth={'token': token})
    received = client.get_received()

    assert len(received) > 0
    assert received[0]['name'] == 'connected'
    assert received[0]['args'][0]['message'] == '✅ Chauffeur connecté'

def test_socketio_driver_location(app, socketio, driver_user):
    """Test émission localisation driver"""
    driver, user = driver_user
    token = create_access_token(identity=user.public_id)

    client = socketio.test_client(app, auth={'token': token})
    client.emit('driver_location', {'latitude': 46.2044, 'longitude': 6.1432})

    # Vérifier broadcast vers company room
    received = client.get_received(namespace=f'/company_{driver.company_id}')
    assert received[0]['name'] == 'driver_location_update'

def test_socketio_unauthorized(app, socketio):
    """Test connexion sans token → refusé"""
    client = socketio.test_client(app, auth={})
    received = client.get_received()

    assert received[0]['name'] == 'unauthorized'

# ... 8-12 tests supplémentaires (rooms, chat, scoping)
```

**Cas de test total**: ~10-15  
**Gain coverage**: SocketIO 0% → 55%

---

## 🎨 FRONTEND: Plan de Tests Détaillé

### 6. Tests Pages (React Testing Library)

#### Fichier: `frontend/src/pages/company/Dashboard/CompanyDashboard.test.jsx`

```javascript
import { render, screen, waitFor } from "@testing-library/react";
import { Provider } from "react-redux";
import { BrowserRouter } from "react-router-dom";
import CompanyDashboard from "./CompanyDashboard";
import companyService from "../../../services/companyService";

jest.mock("../../../services/companyService");
jest.mock("../../../services/companySocket");

const mockStore = {
  auth: { user: { id: 1, role: "company" }, token: "fake-token" },
};

describe("CompanyDashboard", () => {
  it("renders dashboard with bookings", async () => {
    companyService.getMe.mockResolvedValue({ id: 1, name: "Test Transport" });
    companyService.getBookings.mockResolvedValue([
      { id: 1, customer_name: "Patient A", status: "PENDING" },
      { id: 2, customer_name: "Patient B", status: "ASSIGNED" },
    ]);

    render(
      <Provider store={mockStore}>
        <BrowserRouter>
          <CompanyDashboard />
        </BrowserRouter>
      </Provider>
    );

    await waitFor(() => {
      expect(screen.getByText("Patient A")).toBeInTheDocument();
      expect(screen.getByText("Patient B")).toBeInTheDocument();
    });
  });

  it("displays error on fetch failure", async () => {
    companyService.getBookings.mockRejectedValue(new Error("API Error"));

    render(/* ... */);

    await waitFor(() => {
      expect(screen.getByText(/erreur/i)).toBeInTheDocument();
    });
  });

  // ... 5-8 tests supplémentaires (filters, pagination, actions)
});
```

**Pages à tester** (priorité):

- ✅ CompanyDashboard
- ✅ InvoiceList / InvoiceDetail
- ✅ DispatchMonitoring
- ✅ Login / Signup
- DriverDashboard
- ClientReservations

**Cas de test total**: ~40-60 (toutes pages)  
**Gain coverage**: Pages 0% → 65%

---

### 7. Tests Hooks

#### Fichier: `frontend/src/hooks/useAuthToken.test.js`

```javascript
import { renderHook, act } from "@testing-library/react-hooks";
import useAuthToken from "./useAuthToken";

describe("useAuthToken", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("returns null if no token stored", () => {
    const { result } = renderHook(() => useAuthToken());
    expect(result.current.token).toBeNull();
  });

  it("returns token from localStorage", () => {
    localStorage.setItem("authToken", "fake-jwt-token");
    const { result } = renderHook(() => useAuthToken());
    expect(result.current.token).toBe("fake-jwt-token");
  });

  it("refreshes token on 401", async () => {
    // Mock refresh API call
    // Simulate 401 error
    // Assert token refreshed
  });
});
```

**Hooks à tester**:

- useAuthToken
- useCompanySocket
- useDispatchStatus
- useDriver
- useDriverLocation

**Cas de test total**: ~20-30  
**Gain coverage**: Hooks 0% → 80%

---

### 8. Tests Services Frontend

#### Fichier: `frontend/src/services/apiClient.test.js`

```javascript
import axios from "axios";
import MockAdapter from "axios-mock-adapter";
import apiClient, { logoutUser } from "../utils/apiClient";

const mock = new MockAdapter(apiClient);

describe("apiClient", () => {
  afterEach(() => {
    mock.reset();
    localStorage.clear();
  });

  it("adds Authorization header if token present", async () => {
    localStorage.setItem("authToken", "fake-token");
    mock.onGet("/api/test").reply(200, { data: "ok" });

    await apiClient.get("/api/test");

    expect(mock.history.get[0].headers.Authorization).toBe("Bearer fake-token");
  });

  it("handles 401 and calls logoutUser", async () => {
    delete window.location;
    window.location = { href: "" };

    mock.onGet("/api/protected").reply(401);

    await apiClient.get("/api/protected").catch(() => {});

    expect(window.location.href).toBe("/login");
    expect(localStorage.getItem("authToken")).toBeNull();
  });

  it("handles 429 rate limit gracefully", async () => {
    const consoleWarn = jest.spyOn(console, "warn").mockImplementation();
    mock.onGet("/api/test").reply(429);

    await apiClient.get("/api/test").catch(() => {});

    expect(consoleWarn).toHaveBeenCalledWith(
      expect.stringContaining("trop de requêtes")
    );
  });
});
```

**Services à tester**:

- apiClient
- companyService
- invoiceService
- companySocket
- dispatchMonitoringService

**Cas de test total**: ~25-35  
**Gain coverage**: Services 0% → 75%

---

### 9. Tests E2E (Cypress)

#### Fichier: `frontend/cypress/e2e/company-flow.cy.js`

```javascript
describe("Company Flow - Login to Dispatch", () => {
  beforeEach(() => {
    cy.intercept("POST", "/api/auth/login").as("login");
    cy.intercept("GET", "/api/companies/me").as("getCompany");
    cy.intercept("GET", "/api/companies/me/bookings").as("getBookings");
  });

  it("logs in and views dashboard", () => {
    cy.visit("/login");

    cy.get('input[name="email"]').type("company@test.com");
    cy.get('input[name="password"]').type("password123");
    cy.get('button[type="submit"]').click();

    cy.wait("@login").its("response.statusCode").should("eq", 200);
    cy.url().should("include", "/dashboard");

    cy.wait("@getCompany");
    cy.contains("Test Transport"); // Company name
  });

  it("creates booking and assigns driver", () => {
    cy.login("company@test.com", "password123"); // Custom command

    cy.visit("/bookings");
    cy.get('[data-testid="create-booking-btn"]').click();

    cy.get('input[name="customer_name"]').type("Patient Test");
    cy.get('input[name="pickup"]').type("Genève, Rue du Rhône 1");
    cy.get('input[name="dropoff"]').type("Lausanne, CHUV");
    cy.get('input[name="scheduled_time"]').type("2025-10-20T14:00");
    cy.get('button[type="submit"]').click();

    cy.wait("@createBooking").its("response.statusCode").should("eq", 201);
    cy.contains("Réservation créée");

    // Assign driver
    cy.get('[data-testid="booking-1"]').click();
    cy.get('[data-testid="assign-driver-btn"]').click();
    cy.get('[data-testid="driver-select"]').select("Driver A");
    cy.get('[data-testid="confirm-assign"]').click();

    cy.contains("Chauffeur assigné");
  });

  it("triggers dispatch and views result", () => {
    cy.login("company@test.com", "password123");

    cy.visit("/dispatch");
    cy.get('[data-testid="run-dispatch-btn"]').click();

    cy.wait("@runDispatch", { timeout: 10000 });
    cy.contains("Dispatch terminé");
    cy.get('[data-testid="assignments"]').should("have.length.greaterThan", 0);
  });

  // ... 2-3 tests critiques supplémentaires
});
```

**Scénarios E2E prioritaires**:

1. ✅ Login → Dashboard → Bookings (Company)
2. ✅ Create booking → Assign driver
3. ✅ Trigger dispatch → View assignments
4. Generate invoice → Download PDF
5. Driver login → View route → Update status

**Cas de test total**: 5 tests critiques  
**Durée exécution**: ~3-5min

---

## 📊 Estimation Effort & Planning

| Phase                                | Tâches                                                         | Effort (j-h)       | Priorité |
| ------------------------------------ | -------------------------------------------------------------- | ------------------ | -------- |
| **Phase 1: Backend Routes**          | test_routes_auth, test_routes_bookings, test_routes_invoices   | 5j                 | Critique |
| **Phase 2: Backend Services**        | test_service_invoice, test_service_osrm, test_service_dispatch | 4j                 | Critique |
| **Phase 3: Backend Models/Tasks**    | test*models*\*, test_tasks_celery, test_socketio               | 3j                 | Élevée   |
| **Phase 4: Frontend Pages**          | Dashboard, Invoices, Dispatch (RTL)                            | 3j                 | Élevée   |
| **Phase 5: Frontend Hooks/Services** | useAuthToken, apiClient, companySocket                         | 2j                 | Moyenne  |
| **Phase 6: E2E Cypress**             | 5 scénarios critiques                                          | 2j                 | Critique |
| **TOTAL**                            |                                                                | **19 jours-homme** |          |

**Remarque**: Effort peut être réduit à ~12-15j si parallélisation (2 devs) ou focus uniquement critiques.

---

## ✅ Checklist Mise en Place

### Backend (pytest)

- [ ] Installer pytest, pytest-cov, pytest-mock, responses, faker
- [ ] Créer `backend/tests/conftest.py` avec fixtures app/db/test_client
- [ ] Config `backend/pytest.ini` (coverage reports, markers)
- [ ] Créer dossier tests structuré (test_routes/, test_services/, test_models/)
- [ ] Setup DB test (SQLite in-memory ou Postgres test)
- [ ] Exécuter `pytest --cov=backend --cov-report=html`

### Frontend (Jest/RTL)

- [ ] Installer @testing-library/react, @testing-library/jest-dom, axios-mock-adapter
- [ ] Config `frontend/src/setupTests.js` (mocks global, localStorage)
- [ ] Créer `__tests__/` dans chaque dossier (pages, components, hooks)
- [ ] Setup MSW (Mock Service Worker) pour API mocking
- [ ] Exécuter `npm test -- --coverage`

### E2E (Cypress)

- [ ] Installer Cypress: `npm install --save-dev cypress`
- [ ] Init: `npx cypress open`
- [ ] Créer custom commands (`cy.login`, `cy.createBooking`)
- [ ] Config `cypress.config.js` (baseUrl, env vars)
- [ ] Exécuter `npx cypress run`

---

## 🎯 Objectif Final

**Backend**: 70%+ coverage (routes 80%, services 70%, models 85%)  
**Frontend**: 60%+ coverage (pages 65%, hooks 80%, services 75%)  
**E2E**: 5 tests critiques couvrant happy paths principaux

**Impact**:

- Réduction bugs production -60%
- Refactorings sûrs (tests régression)
- CI/CD fiable (tests automatiques)
- Documentation vivante (tests = specs)

---

_Document généré le 15 octobre 2025. Plan à ajuster selon ressources disponibles._
