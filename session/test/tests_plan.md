# 🧪 Plan de Tests Complet ATMR

## 📋 Vue d'ensemble

Ce document définit la stratégie de tests pour les trois composants de l'application ATMR :

- **Backend** (pytest + fixtures + mocks)
- **Frontend** (React Testing Library + Cypress E2E)
- **Mobile** (Jest + React Native Testing Library)

---

## 🎯 Objectifs de Couverture

| Composant                         | Cible       | Priorité | Délai       |
| --------------------------------- | ----------- | -------- | ----------- |
| Backend routes critiques          | ≥70%        | P0       | Semaine 1-2 |
| Backend services (dispatch, OSRM) | ≥60%        | P1       | Semaine 2   |
| Frontend composants UI            | ≥60%        | P1       | Semaine 2   |
| Frontend E2E (user flows)         | 5 scénarios | P1       | Semaine 3   |
| Mobile composants                 | ≥50%        | P2       | Semaine 4   |
| Mobile services                   | ≥50%        | P2       | Semaine 4   |

---

## 🔧 Backend : pytest + fixtures + mocks

### Configuration initiale

```python
# backend/conftest.py
import pytest
from app import create_app
from ext import db as _db
from models import User, Company, Client, Booking, Driver

@pytest.fixture(scope='session')
def app():
    """Crée une instance Flask en mode test."""
    app = create_app('testing')
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        'RATELIMIT_ENABLED': False,
    })
    return app

@pytest.fixture(scope='function')
def db(app):
    """Crée une DB propre pour chaque test."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()

@pytest.fixture
def client(app, db):
    """Client de test Flask."""
    return app.test_client()

@pytest.fixture
def auth_headers(client, db):
    """Génère un token JWT valide pour un utilisateur test."""
    from models import User, UserRole
    from ext import bcrypt
    user = User(
        username='testuser',
        email='test@example.com',
        role=UserRole.company,
        public_id='test-uuid-1234'
    )
    user.password = bcrypt.generate_password_hash('password123').decode('utf-8')
    db.session.add(user)
    db.session.commit()

    response = client.post('/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    token = response.get_json()['token']
    return {'Authorization': f'Bearer {token}'}
```

### Tests prioritaires (P0/P1)

#### 1. **Auth** (`tests/test_auth.py`)

```python
def test_login_success(client, db):
    """Login avec credentials valides renvoie un token."""
    # Créer utilisateur
    # POST /api/auth/login
    # Assert 200, token présent, refresh_token présent

def test_login_invalid_password(client, db):
    """Login avec mauvais mot de passe renvoie 401."""

def test_refresh_token(client, auth_headers):
    """Refresh token valide génère un nouveau token d'accès."""

def test_protected_route_no_token(client):
    """Accès à route protégée sans token renvoie 401."""
```

#### 2. **Bookings** (`tests/test_bookings.py`)

```python
def test_create_booking(client, auth_headers, db):
    """Création d'une réservation client renvoie 201 avec ID."""

def test_create_round_trip(client, auth_headers, db):
    """is_round_trip=true crée 2 bookings liés."""

def test_update_booking_status(client, auth_headers, db):
    """Mise à jour statut PENDING -> CONFIRMED."""

def test_cancel_booking(client, auth_headers, db):
    """Annulation d'une réservation met status=CANCELLED."""

def test_assign_driver(client, auth_headers, db):
    """Assigner un chauffeur à une réservation met driver_id."""
```

#### 3. **Dispatch** (`tests/test_dispatch.py`)

```python
@pytest.fixture
def mock_osrm(monkeypatch):
    """Mock OSRM pour éviter appels réseau."""
    def fake_table(*args, **kwargs):
        # Retourne une matrice 3x3 factice
        return {
            'code': 'Ok',
            'durations': [[0, 600, 1200], [600, 0, 800], [1200, 800, 0]]
        }
    monkeypatch.setattr('services.osrm_client._table', fake_table)
    return fake_table

def test_dispatch_run_creates_assignments(client, auth_headers, mock_osrm, db):
    """POST /api/company_dispatch/run crée des assignments."""
    # Créer 2 bookings + 1 driver
    # Trigger dispatch
    # Assert assignments créés, dispatch_run_id présent

def test_dispatch_unassigned_bookings(client, auth_headers, mock_osrm, db):
    """Bookings sans chauffeur disponible restent unassigned."""

def test_dispatch_retry_on_osrm_timeout(client, auth_headers, monkeypatch, db):
    """OSRM timeout déclenche fallback haversine."""
```

#### 4. **Invoices** (`tests/test_invoices.py`)

```python
def test_generate_invoice(client, auth_headers, db):
    """Génération facture pour un client avec bookings completed."""

def test_invoice_line_tva_calculation(db):
    """Calcul TVA 8.1% sur montant HT."""

def test_qrbill_generation(db):
    """QR-bill généré contient IBAN, référence, montant."""

def test_invoice_sequence_auto_increment(db):
    """Numéro facture s'incrémente automatiquement."""
```

#### 5. **OSRM Client** (`tests/test_osrm_client.py`)

```python
def test_osrm_table_success(monkeypatch):
    """OSRM table renvoie matrice de durées."""

def test_osrm_timeout_fallback_haversine(monkeypatch):
    """Timeout OSRM déclenche calcul haversine."""

def test_osrm_cache_hit(monkeypatch, mocker):
    """Cache Redis retourne résultat sans appel HTTP."""

def test_osrm_cache_miss_then_hit(monkeypatch, mocker):
    """Premier appel cache miss, second hit."""
```

#### 6. **Celery Tasks** (`tests/test_celery_tasks.py`)

```python
@pytest.fixture
def celery_app(app):
    """Celery en mode eager (exécution synchrone)."""
    from celery_app import celery
    celery.conf.update(task_always_eager=True, task_eager_propagates=True)
    return celery

def test_dispatch_task_success(celery_app, db):
    """Task run_dispatch_task retourne résultat avec assignments."""

def test_dispatch_task_retry_on_db_error(celery_app, monkeypatch):
    """Erreur DB déclenche retry avec backoff."""

def test_autorun_tick_triggers_dispatch(celery_app, db):
    """autorun_tick lance dispatch pour companies avec dispatch_enabled=true."""
```

### Stratégie de Mocks

| Service      | Bibliothèque            | Raison                                 |
| ------------ | ----------------------- | -------------------------------------- |
| OSRM HTTP    | `monkeypatch`           | Éviter appels réseau, tester fallbacks |
| Celery       | `task_always_eager`     | Exécution synchrone en tests           |
| Redis        | `fakeredis`             | Éviter dépendance Redis externe        |
| Google Maps  | `responses`             | Mock HTTP geocode/distance             |
| Socket.IO    | `socketio.test_client`  | Émission/réception événements          |
| Email (SMTP) | `monkeypatch` mail.send | Éviter envoi réel                      |

### Commandes

```bash
# Installation
pip install pytest pytest-flask pytest-cov fakeredis responses

# Exécution
pytest -v --cov=backend --cov-report=html

# Tests spécifiques
pytest tests/test_auth.py -k login
pytest tests/test_dispatch.py -v -s  # verbose + print
```

---

## ⚛️ Frontend : React Testing Library + Cypress

### Configuration initiale

```javascript
// frontend/src/setupTests.js
import "@testing-library/jest-dom";
import { server } from "./mocks/server";

// Mock API avec MSW (Mock Service Worker)
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

```javascript
// frontend/src/mocks/server.js
import { setupServer } from "msw/node";
import { rest } from "msw";

const handlers = [
  rest.post("/api/auth/login", (req, res, ctx) => {
    return res(
      ctx.json({
        token: "fake-token",
        refresh_token: "fake-refresh",
      })
    );
  }),
  rest.get("/api/bookings", (req, res, ctx) => {
    return res(
      ctx.json({
        bookings: [{ id: 1, customer_name: "Test Client" }],
      })
    );
  }),
];

export const server = setupServer(...handlers);
```

### Tests unitaires (P1)

#### 1. **Login** (`src/pages/Auth/Login.test.jsx`)

```jsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import Login from "./Login";

test("affiche formulaire de connexion", () => {
  render(
    <BrowserRouter>
      <Login />
    </BrowserRouter>
  );
  expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/mot de passe/i)).toBeInTheDocument();
});

test("soumet formulaire et redirige vers dashboard", async () => {
  const { container } = render(
    <BrowserRouter>
      <Login />
    </BrowserRouter>
  );

  fireEvent.change(screen.getByLabelText(/email/i), {
    target: { value: "test@example.com" },
  });
  fireEvent.change(screen.getByLabelText(/mot de passe/i), {
    target: { value: "password123" },
  });

  fireEvent.click(screen.getByRole("button", { name: /connexion/i }));

  await waitFor(() => {
    expect(localStorage.getItem("authToken")).toBe("fake-token");
  });
});
```

#### 2. **Booking Form** (`src/pages/company/Reservations/components/NewBookingModal.test.jsx`)

```jsx
test("validation champs obligatoires", async () => {
  render(<NewBookingModal />);

  fireEvent.click(screen.getByRole("button", { name: /créer/i }));

  await waitFor(() => {
    expect(screen.getByText(/customer_name requis/i)).toBeInTheDocument();
  });
});

test("création réservation aller-retour", async () => {
  render(<NewBookingModal />);

  fireEvent.change(screen.getByLabelText(/nom client/i), {
    target: { value: "Jean Dupont" },
  });
  fireEvent.click(screen.getByLabelText(/aller-retour/i));

  fireEvent.click(screen.getByRole("button", { name: /créer/i }));

  await waitFor(() => {
    // Assert 2 bookings créés
  });
});
```

### Tests E2E (Cypress) - P1

```bash
# Installation
npm install --save-dev cypress @testing-library/cypress

# Configuration
npx cypress open
```

#### Scénarios critiques

```javascript
// cypress/e2e/user-flow.cy.js
describe("User Flow Complet", () => {
  beforeEach(() => {
    cy.visit("http://localhost:3000");
  });

  it("Login → Dashboard → Créer booking → Dispatch", () => {
    // 1. Login
    cy.get('input[name="email"]').type("company@example.com");
    cy.get('input[name="password"]').type("password123");
    cy.get('button[type="submit"]').click();

    // 2. Dashboard
    cy.url().should("include", "/company/dashboard");
    cy.contains("Tableau de bord").should("be.visible");

    // 3. Créer booking
    cy.contains("Réservations").click();
    cy.contains("Nouvelle réservation").click();
    cy.get('input[name="customer_name"]').type("Test Client");
    cy.get('input[name="pickup_location"]').type("Genève");
    cy.get('input[name="dropoff_location"]').type("Lausanne");
    cy.get("button").contains("Créer").click();

    // 4. Dispatch
    cy.contains("Dispatch").click();
    cy.get("button").contains("Lancer dispatch").click();
    cy.contains("Assignments créés", { timeout: 10000 }).should("be.visible");
  });

  it("Login → Facturation → Générer facture", () => {
    cy.login("company@example.com", "password123");
    cy.visit("/company/invoices");
    cy.get("button").contains("Nouvelle facture").click();
    cy.get('select[name="client_id"]').select("1");
    cy.get("button").contains("Générer").click();
    cy.contains("Facture créée").should("be.visible");
  });
});
```

### Commandes

```bash
# Tests unitaires
npm test -- --coverage

# E2E Cypress (UI)
npx cypress open

# E2E headless (CI)
npx cypress run
```

---

## 📱 Mobile : Jest + React Native Testing Library

### Configuration

```javascript
// mobile/driver-app/jest.config.js
module.exports = {
  preset: "jest-expo",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],
  transformIgnorePatterns: [
    "node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|@unimodules/.*|unimodules|sentry-expo|native-base|react-native-svg)",
  ],
  collectCoverageFrom: [
    "app/**/*.{ts,tsx}",
    "components/**/*.{ts,tsx}",
    "services/**/*.ts",
    "!**/*.d.ts",
  ],
};
```

```javascript
// mobile/driver-app/jest.setup.js
import "@testing-library/jest-native/extend-expect";

// Mock AsyncStorage
jest.mock("@react-native-async-storage/async-storage", () => ({
  setItem: jest.fn(),
  getItem: jest.fn(),
  removeItem: jest.fn(),
}));

// Mock expo-location
jest.mock("expo-location", () => ({
  requestForegroundPermissionsAsync: jest.fn(),
  getCurrentPositionAsync: jest.fn(),
  watchPositionAsync: jest.fn(),
}));
```

### Tests prioritaires

#### 1. **Components** (`components/__tests__/MissionCard.test.tsx`)

```typescript
import { render, screen } from "@testing-library/react-native";
import MissionCard from "../dashboard/MissionCard";

test("affiche nom client et adresse pickup", () => {
  const mission = {
    id: 1,
    customer_name: "Jean Dupont",
    pickup_location: "Genève",
    dropoff_location: "Lausanne",
  };

  render(<MissionCard mission={mission} />);

  expect(screen.getByText("Jean Dupont")).toBeTruthy();
  expect(screen.getByText(/Genève/i)).toBeTruthy();
});
```

#### 2. **Hooks** (`hooks/__tests__/useAuth.test.tsx`)

```typescript
import { renderHook, act } from "@testing-library/react-hooks";
import useAuth from "../useAuth";

test("login stocke token dans AsyncStorage", async () => {
  const { result } = renderHook(() => useAuth());

  await act(async () => {
    await result.current.login("driver@example.com", "password");
  });

  expect(AsyncStorage.setItem).toHaveBeenCalledWith(
    "authToken",
    expect.any(String)
  );
});
```

#### 3. **Services** (`services/__tests__/api.test.ts`)

```typescript
import api from "../api";

test("refresh token automatique sur 401", async () => {
  // Mock fetch 401 puis 200 après refresh
  global.fetch = jest
    .fn()
    .mockResolvedValueOnce({ status: 401 })
    .mockResolvedValueOnce({
      status: 200,
      json: async () => ({ token: "new-token" }),
    });

  const response = await api.get("/driver/missions");

  expect(global.fetch).toHaveBeenCalledTimes(2);
});
```

### Commandes

```bash
# Tests
npm test -- --coverage

# Watch mode
npm test -- --watch

# Tests spécifiques
npm test -- MissionCard.test.tsx
```

---

## 🎭 Stratégie de Mocks Globale

### Backend

```python
# Mocks par service
OSRM HTTP       → monkeypatch / responses
Google Maps     → responses (mock JSON)
Redis           → fakeredis
Celery          → task_always_eager=True
Email SMTP      → monkeypatch mail.send
SocketIO        → socketio.test_client()
```

### Frontend

```javascript
// MSW (Mock Service Worker)
API REST        → msw handlers
SocketIO        → socket.io-mock
LocalStorage    → jest.spyOn(Storage.prototype)
Geolocation     → navigator.geolocation mock
```

### Mobile

```typescript
// Jest mocks
AsyncStorage    → @react-native-async-storage mock
expo-location   → mock requestPermissions, getCurrentPosition
axios           → jest.mock('axios')
socket.io       → socket.io-mock
```

---

## 📊 Métriques de Succès

| Métrique                | Cible      | Outil             |
| ----------------------- | ---------- | ----------------- |
| **Couverture backend**  | ≥70%       | pytest-cov        |
| **Couverture frontend** | ≥60%       | jest --coverage   |
| **Couverture mobile**   | ≥50%       | jest --coverage   |
| **E2E scénarios**       | 5 passants | Cypress Dashboard |
| **Tests flaky**         | <5%        | CI stats          |
| **Temps exécution CI**  | <10min     | GitHub Actions    |

---

## 🚀 Intégration CI/CD

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r backend/requirements.txt
      - run: pip install pytest pytest-cov
      - run: cd backend && pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "18"
      - run: cd frontend && npm ci
      - run: cd frontend && npm test -- --coverage

  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cypress-io/github-action@v6
        with:
          start: npm start
          wait-on: "http://localhost:3000"
```

---

## 📝 Checklist Avant Merge

- [ ] Tests unitaires backend ≥70% sur fichiers modifiés
- [ ] Tests frontend ajoutés pour nouveaux composants
- [ ] E2E Cypress passent (si user flow modifié)
- [ ] Aucun test flaky (réexécution 3x réussie)
- [ ] Coverage ne baisse pas (comparaison main)
- [ ] CI green (tous jobs passent)
- [ ] Mocks documentés (README ou docstrings)

---

**Prochaine étape** : Implémenter tests P0 (auth, bookings, dispatch) en semaine 1-2.
