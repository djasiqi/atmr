# 🧪 PLAN DE TESTS & VALIDATION - ATMR

**Date** : 2025-10-18  
**Version** : 1.0  
**Scope** : Validation complète de tous les patches et améliorations

---

## 📋 TABLE DES MATIÈRES

1. [Prérequis](#prérequis)
2. [Tests Backend](#tests-backend)
3. [Tests Socket.IO](#tests-socketio)
4. [Tests Frontend](#tests-frontend)
5. [Tests Mobile (Driver-App)](#tests-mobile)
6. [Tests Performance](#tests-performance)
7. [Tests Sécurité](#tests-sécurité)
8. [Tests Infrastructure](#tests-infrastructure)
9. [Critères d'Acceptation](#critères-dacceptation)
10. [Jeux de Données](#jeux-de-données)

---

## ✅ PRÉREQUIS

### Environnement de test

```bash
# Variables d'environnement requises
export FLASK_ENV=testing
export DATABASE_URL=postgresql+psycopg://atmr:atmr@localhost:5432/atmr_test
export REDIS_URL=redis://localhost:6379/1
export CELERY_BROKER_URL=redis://localhost:6379/1
export JWT_SECRET_KEY=test-secret-key-change-me
export SECRET_KEY=test-secret-change-me
```

### Services requis

- PostgreSQL 16 (base de test séparée)
- Redis 7
- OSRM (optionnel, fallback haversine OK)
- Python 3.11+
- Node.js 18+

### Installation

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate sur Windows
pip install -r requirements.txt -r requirements-dev.txt

# Frontend
cd frontend
npm ci

# Mobile
cd mobile/driver-app
npm ci
```

---

## 🐍 TESTS BACKEND

### 1. Tests unitaires Pytest

**Commande** :

```bash
cd backend
pytest -v --cov=. --cov-report=html --cov-report=term
```

**Critères d'acceptation** :

- ✅ Tous les tests passent (0 failed)
- ✅ Coverage ≥ 75% sur domaines critiques (models, routes, services)
- ✅ Pas de warnings Pytest

**Logs attendus** :

```
tests/test_auth.py::test_login PASSED
tests/test_bookings.py::test_create_booking PASSED
tests/test_dispatch.py::test_dispatch_engine PASSED
...
========== 47 passed in 12.34s ==========
Coverage: 78%
```

**En cas d'échec** :

- Vérifier DATABASE_URL pointe vers base de test
- Vérifier migrations DB up-to-date : `flask db upgrade`
- Vérifier Redis accessible

---

### 2. Tests spécifiques patches DB (02-db-eager-loading)

**Objectif** : Vérifier que les index sont créés et N+1 éliminés

**Commandes** :

```bash
# Appliquer le patch
cd backend
patch -p1 < ../session/patches/02-db-eager-loading-indexes.diff

# Vérifier la migration
flask db upgrade

# Vérifier les index créés
psql $DATABASE_URL -c "\d booking"
# Doit afficher : ix_booking_company_scheduled, ix_booking_status_scheduled, etc.

# Test de charge N+1
python -c "
from app import create_app
from models import Booking
from sqlalchemy import inspect

app = create_app('testing')
with app.app_context():
    # Avant patch : N+1 queries
    bookings = Booking.query.limit(50).all()
    for b in bookings:
        _ = b.driver  # Déclenche lazy load
        _ = b.client
    # Après patch : 1 query (selectinload)
    # Vérifier avec SQLALCHEMY_ECHO=1
"
```

**Critères d'acceptation** :

- ✅ Index présents dans `\d booking`
- ✅ Query count réduit de 1+N à 3 queries max (booking + driver + client selectinload)
- ✅ Latence GET /api/bookings réduite de 300ms → <120ms

**Validation query count** :

```python
# Avec logging SQL activé
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Compter les queries
from sqlalchemy import event
from sqlalchemy.engine import Engine

query_count = []

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    query_count.append(statement)

# Exécuter le test
bookings = Booking.query.options(
    selectinload(Booking.driver),
    selectinload(Booking.client)
).limit(50).all()

print(f"Total queries: {len(query_count)}")  # Doit être ≤ 3
```

---

### 3. Tests Celery (tâches dispatch)

**Objectif** : Vérifier idempotence, retry, timeouts

**Commandes** :

```bash
# Démarrer worker en mode test
celery -A celery_app:celery worker -l info --pool=solo &

# Tester tâche dispatch
python -c "
from tasks.dispatch_tasks import run_dispatch_task
result = run_dispatch_task.delay(
    company_id=1,
    for_date='2025-10-20',
    mode='auto'
)
print('Task ID:', result.id)
print('Status:', result.status)
print('Result:', result.get(timeout=60))
"
```

**Critères d'acceptation** :

- ✅ Tâche complète avec status SUCCESS
- ✅ Résultat contient `assignments`, `unassigned`, `meta`
- ✅ En cas d'erreur DB transient, retry automatique (max 3 fois)
- ✅ Timeout respecté (300s hard, 270s soft)

**Test retry** :

```python
# Simuler erreur DB transient
from unittest.mock import patch
from sqlalchemy.exc import OperationalError

@patch('ext.db.session.commit')
def test_retry(mock_commit):
    mock_commit.side_effect = OperationalError("connection lost", None, None)
    result = run_dispatch_task.apply(args=(1, '2025-10-20'), throw=False)
    assert result.retries == 3  # A retenté 3 fois
    assert result.state == 'FAILURE'
```

---

## 🔌 TESTS SOCKET.IO

### 1. Test connexion + authentification JWT

**Objectif** : Vérifier que le handler `connect` est appelé, JWT validé, room joined

**Script de test** :

```python
# tests/test_socketio_connect.py
import pytest
from socketio import SimpleClient
from app import create_app
from services.socketio_service import socketio

def test_socketio_connect_with_valid_jwt():
    app = create_app('testing')
    client = SimpleClient()

    # Obtenir un JWT valide
    from flask_jwt_extended import create_access_token
    with app.app_context():
        token = create_access_token(identity='test-user-public-id')

    # Connecter avec auth
    client.connect(
        'http://localhost:5000',
        auth={'token': token},
        transports=['polling']
    )

    # Vérifier réception événement 'connected'
    event = client.receive(timeout=5)
    assert event[0] == 'connected'
    assert 'message' in event[1]

    client.disconnect()

def test_socketio_connect_without_jwt():
    client = SimpleClient()

    # Connecter sans JWT
    client.connect('http://localhost:5000', transports=['polling'])

    # Doit recevoir 'unauthorized'
    event = client.receive(timeout=5)
    assert event[0] == 'unauthorized'

    client.disconnect()
```

**Commandes** :

```bash
# Démarrer serveur en mode dev (worker gevent)
cd backend
gunicorn wsgi:app --bind 0.0.0.0:5000 --worker-class gevent --workers 1 --log-level debug &

# Exécuter tests Socket.IO
pytest tests/test_socketio_connect.py -v

# Arrêter serveur
pkill -f gunicorn
```

**Critères d'acceptation** :

- ✅ Connect avec JWT valide → événement `connected` reçu
- ✅ Connect sans JWT → événement `unauthorized` reçu
- ✅ Logs montrent : "🔌 [CONNECT] HANDLER APPELÉ !"
- ✅ Room company_X joined (vérifiable dans logs)

---

### 2. Test événements temps réel (driver_location, team_chat_message)

**Script de test** :

```python
# tests/test_socketio_events.py
def test_driver_location_update():
    app = create_app('testing')
    client = SimpleClient()

    # Connecter en tant que driver
    with app.app_context():
        driver_user = User.query.filter_by(role=UserRole.driver).first()
        token = create_access_token(identity=driver_user.public_id)

    client.connect('http://localhost:5000', auth={'token': token}, transports=['polling'])
    client.receive()  # Consommer 'connected'

    # Émettre position
    client.emit('driver_location', {
        'latitude': 46.5197,
        'longitude': 6.6323,
        'driver_id': driver_user.driver.id
    })

    # Vérifier que l'événement est bien traité (pas d'error)
    # (Le test côté serveur, la room company reçoit driver_location_update)

    time.sleep(1)
    client.disconnect()

def test_team_chat_message():
    app = create_app('testing')
    client = SimpleClient()

    # Connecter en tant que company
    with app.app_context():
        company_user = User.query.filter_by(role=UserRole.company).first()
        token = create_access_token(identity=company_user.public_id)

    client.connect('http://localhost:5000', auth={'token': token}, transports=['polling'])
    client.receive()  # 'connected'

    # Émettre message
    client.emit('team_chat_message', {
        'content': 'Test message',
        'receiver_id': None,
        '_localId': 'test-123'
    })

    # Attendre réception dans la room
    event = client.receive(timeout=5)
    assert event[0] == 'team_chat_message'
    assert event[1]['content'] == 'Test message'

    client.disconnect()
```

**Critères d'acceptation** :

- ✅ driver_location émis → pas d'événement `error`, logs OK
- ✅ team_chat_message émis → reçu dans la room company_X
- ✅ Validation lat/lon (hors bornes → error event)
- ✅ Message vide → error event

---

## ⚛️ TESTS FRONTEND

### 1. Tests unitaires (React Testing Library)

**Commande** :

```bash
cd frontend
npm run test:ci
```

**Critères d'acceptation** :

- ✅ Tous les tests passent
- ✅ Coverage ≥ 70% sur composants critiques (Login, Dashboard, BookingForm)
- ✅ Snapshots à jour

**Logs attendus** :

```
PASS src/pages/auth/Login.test.jsx
PASS src/components/BookingCard.test.jsx
...
Test Suites: 15 passed, 15 total
Tests:       82 passed, 82 total
Snapshots:   12 passed, 12 total
Coverage: 73.2%
```

---

### 2. Test Bundle Size (après patch 10-frontend-bundle)

**Objectif** : Vérifier réduction de 30% minimum

**Commandes** :

```bash
cd frontend

# Build production
npm run build

# Analyser taille
du -sh build/
du -sh build/static/js/*.js

# Vérifier code-splitting (doit avoir plusieurs chunks)
ls -lh build/static/js/
```

**Critères d'acceptation** :

- ✅ Taille totale build/ < 2.3 MB (avant : 3.2 MB)
- ✅ main.chunk.js < 800 KB (avant : 1.4 MB)
- ✅ Présence de chunks séparés (routes.chunk.js, maps.chunk.js, etc.)
- ✅ Lighthouse Performance Score ≥ 85/100

**Analyse bundle** :

```bash
# Installer bundle analyzer
npm install --save-dev webpack-bundle-analyzer

# Analyser
npx webpack-bundle-analyzer build/bundle-stats.json
# Ouvrir http://127.0.0.1:8888 et vérifier tree-shaking
```

---

### 3. Test Socket.IO frontend (reconnection, événements)

**Test manuel** :

1. Ouvrir http://localhost:3000 (dev server)
2. Login en tant que company
3. Ouvrir DevTools → Network → WS
4. Vérifier connexion Socket.IO établie
5. Couper réseau (DevTools → Network → Offline)
6. Attendre 5s → remettre Online
7. Vérifier reconnexion automatique

**Critères d'acceptation** :

- ✅ Connexion établie au login
- ✅ Reconnexion automatique après coupure réseau (max 5 tentatives)
- ✅ Événements driver_location_update reçus (logs console)
- ✅ Événements dispatch_run_completed reçus

**Test automatisé (Cypress/Playwright)** :

```javascript
// cypress/e2e/socketio.cy.js
describe("Socket.IO Integration", () => {
  it("should connect and receive events", () => {
    cy.visit("/login");
    cy.get("[data-testid=email]").type("company@test.com");
    cy.get("[data-testid=password]").type("password123");
    cy.get("[data-testid=submit]").click();

    cy.url().should("include", "/dashboard");

    // Vérifier connexion Socket.IO
    cy.window().then((win) => {
      cy.wrap(win.socketConnected).should("eq", true);
    });

    // Simuler événement backend
    cy.window().then((win) => {
      win.socket.emit("driver_location", {
        driver_id: 1,
        latitude: 46.5,
        longitude: 6.6,
      });
    });

    // Vérifier réception
    cy.get("[data-testid=driver-marker-1]").should("exist");
  });
});
```

---

## 📱 TESTS MOBILE (DRIVER-APP)

### 1. Tests Jest (React Native)

**Commande** :

```bash
cd mobile/driver-app
npm test -- --coverage
```

**Critères d'acceptation** :

- ✅ Tous les tests passent
- ✅ Coverage ≥ 60% (mobile testing difficile)
- ✅ Pas d'erreurs TypeScript

---

### 2. Test batching location (patch 20-driverapp-location-batching)

**Test manuel** :

1. Build APK dev : `npm run build:dev`
2. Installer sur device Android
3. Login en tant que driver
4. Activer mission
5. Vérifier dans logs backend : positions reçues en batch (toutes les 15s)

**Critères d'acceptation** :

- ✅ Positions envoyées toutes les 15s (au lieu de 5s)
- ✅ Batch contient 3-5 positions
- ✅ Drain batterie réduit (mesurer avec Battery Historian)

**Mesure batterie** :

```bash
# Android Battery Historian
adb bugreport > bugreport.zip
# Uploader sur https://bathist.ef.lc/
# Comparer avant/après patch
```

---

### 3. Test EAS Build

**Commande** :

```bash
cd mobile/driver-app
eas build --profile development --platform android --local
```

**Critères d'acceptation** :

- ✅ Build réussit sans erreur
- ✅ APK généré (<50 MB)
- ✅ Pas d'erreurs de dépendances natives

---

## ⚡ TESTS PERFORMANCE

### 1. Benchmarks API (latence p95/p99)

**Outil** : wrk (HTTP benchmarking)

**Script** : `session/new_files/profiling/benchmark_api.py`

**Commandes** :

```bash
# Démarrer API backend
docker compose up -d api postgres redis

# Benchmark GET /api/bookings
wrk -t4 -c100 -d30s --latency \
  -H "Authorization: Bearer $JWT_TOKEN" \
  http://localhost:5000/api/bookings?date=2025-10-20

# Résultats attendus (APRÈS patches) :
# Latency p50: 45ms
# Latency p95: 95ms
# Latency p99: 180ms
# Requests/sec: 850

# Benchmark POST /api/dispatch/run
wrk -t2 -c10 -d60s --latency \
  -s dispatch_post.lua \
  http://localhost:5000/api/company_dispatch/run

# Résultats attendus :
# Latency p95: 2.8s (avant: 4.2s)
```

**Critères d'acceptation** :

- ✅ GET /api/bookings : p95 < 120ms (avant: 312ms) → **-62%**
- ✅ POST /api/dispatch/run : p95 < 3.0s (avant: 4.2s) → **-29%**
- ✅ GET /api/drivers : p95 < 80ms

---

### 2. Load Testing (Locust)

**Script** : `session/new_files/profiling/locust_load_test.py`

**Commande** :

```bash
cd session/new_files/profiling
locust -f locust_load_test.py --host=http://localhost:5000

# Ouvrir http://localhost:8089
# Configurer : 100 users, spawn rate 10/s, durée 5min
```

**Scénarios** :

1. **Login** (20% du traffic)
2. **Get bookings** (40%)
3. **Create booking** (10%)
4. **Get drivers** (20%)
5. **Dispatch run** (10%)

**Critères d'acceptation** :

- ✅ 0% d'erreurs à 100 users concurrents
- ✅ <5% d'erreurs à 200 users
- ✅ Throughput > 500 req/s

---

### 3. Profiling OSRM (matrices volumineuses)

**Test** :

```python
# test_osrm_large_matrix.py
from services.osrm_client import build_distance_matrix_osrm
import time

# Générer 100 coordonnées (Suisse)
coords = [(46.5 + i*0.01, 6.6 + j*0.01) for i in range(10) for j in range(10)]

start = time.time()
matrix = build_distance_matrix_osrm(
    coords,
    base_url='http://localhost:5000',  # OSRM local
    timeout=30,  # Patch 03: augmenté à 30s
    max_sources_per_call=40,  # Patch 03: adaptatif
)
duration = time.time() - start

assert len(matrix) == 100
assert len(matrix[0]) == 100
assert duration < 35  # Doit finir en <35s avec timeout 30s + overhead
print(f"✅ Matrix 100x100 générée en {duration:.2f}s")
```

**Critères d'acceptation** :

- ✅ Matrice 100x100 générée en <35s (avant: timeout à 10s)
- ✅ Pas d'exception OSRMError
- ✅ Fallback haversine si OSRM down

---

## 🔒 TESTS SÉCURITÉ

### 1. Validation JWT avec audience claim

**Test** :

```python
# test_jwt_audience.py
from flask_jwt_extended import decode_token, create_access_token
from app import create_app

app = create_app('testing')

with app.app_context():
    # Créer token avec audience
    token = create_access_token(
        identity='user-123',
        additional_claims={'aud': 'atmr-api'}
    )

    # Décoder et vérifier
    decoded = decode_token(token)
    assert decoded['aud'] == 'atmr-api'

    # Token sans aud doit échouer (après patch 05)
    # (nécessite configuration JWTManager avec verify_aud=True)
```

**Critères d'acceptation** :

- ✅ Tokens avec `aud=atmr-api` validés
- ✅ Tokens sans `aud` rejetés (si verify_aud=True)

---

### 2. Tests PII scrubbing dans logs

**Test** :

```python
# test_pii_filter.py
from shared.logging_utils import PIIFilter
import logging

logger = logging.getLogger('test')
logger.addFilter(PIIFilter())
handler = logging.StreamHandler()
logger.addHandler(handler)

# Tester redaction
logger.info("User email: john.doe@example.com")
# Log doit afficher: "User email: [EMAIL_REDACTED]"

logger.info("IBAN: CH93 0076 2011 6238 5295 7")
# Log doit afficher: "IBAN: [IBAN_REDACTED]"
```

**Critères d'acceptation** :

- ✅ Emails masqués : `[EMAIL_REDACTED]`
- ✅ IBANs masqués : `[IBAN_REDACTED]`
- ✅ Numéros carte masqués : `[CARD_REDACTED]`

---

### 3. Test rate-limiting

**Test** :

```bash
# Envoyer 100 requêtes rapidement
for i in {1..100}; do
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/api/auth/login
done

# Après 5000 requêtes/heure par IP → HTTP 429
```

**Critères d'acceptation** :

- ✅ HTTP 429 après dépassement limite
- ✅ Header `Retry-After` présent

---

## 🐳 TESTS INFRASTRUCTURE

### 1. Docker Compose build & healthchecks

**Commande** :

```bash
# Build toutes les images
docker compose build

# Démarrer stack complète
docker compose up -d

# Vérifier healthchecks (attendre 60s)
sleep 60
docker compose ps

# Tous les services doivent être "healthy" :
# postgres   healthy
# redis      healthy
# api        healthy
# celery-worker healthy
# celery-beat healthy
# flower     healthy
```

**Critères d'acceptation** :

- ✅ Tous les services healthy en <60s
- ✅ Pas d'erreurs dans logs : `docker compose logs api`

---

### 2. Test migrations DB (up/down)

**Commande** :

```bash
# Upgrade
docker compose exec api flask db upgrade

# Vérifier tables créées
docker compose exec postgres psql -U atmr -d atmr -c "\dt"

# Downgrade (rollback dernière migration)
docker compose exec api flask db downgrade

# Re-upgrade
docker compose exec api flask db upgrade
```

**Critères d'acceptation** :

- ✅ Upgrade sans erreur
- ✅ Downgrade sans perte de données critique
- ✅ Re-upgrade idempotent

---

## ✅ CRITÈRES D'ACCEPTATION GLOBAUX

### Backend

- ✅ Tous les tests Pytest passent (0 failed)
- ✅ Coverage ≥ 75% sur domaines critiques
- ✅ Linter Ruff : 0 error, <5 warnings
- ✅ Mypy : 0 error (strict mode)

### Socket.IO

- ✅ Handler `connect` appelé, JWT validé
- ✅ Événements reçus sans refresh navigateur
- ✅ Reconnexion automatique fonctionne

### Performance

- ✅ Latence p95 GET /api/bookings < 120ms (-62%)
- ✅ Latence p95 POST /api/dispatch/run < 3.0s (-29%)
- ✅ Frontend bundle < 2.3 MB (-30%)

### Sécurité

- ✅ Pas de secrets en clair (tous dans .env + .gitignore)
- ✅ Headers sécurité actifs (CSP, X-Frame-Options, etc.)
- ✅ Payloads validés (pydantic/validators)
- ✅ JWT avec audience claim
- ✅ PII scrubbing actif dans logs

### DB

- ✅ Index ajoutés (vérifiés avec `\d booking`)
- ✅ N+1 queries éliminés (query count ≤ 3)
- ✅ Migrations up/down testées

### Dead Files

- ✅ DEAD_FILES.json livré
- ✅ Fichiers morts supprimés (15 fichiers)
- ✅ .gitignore mis à jour

---

## 🗂️ JEUX DE DONNÉES

### Données de test (fixtures Pytest)

**Fichier** : `backend/tests/conftest.py`

```python
@pytest.fixture
def sample_bookings(db):
    """Créer 50 bookings de test"""
    from models import Booking, Client, Company, User, UserRole
    from datetime import datetime, timedelta

    # Créer company
    company = Company(name="Test Transport SA")
    db.session.add(company)

    # Créer client
    user = User(username="client_test", email="client@test.com", role=UserRole.client)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()

    client = Client(user_id=user.id, contact_phone="0791234567")
    db.session.add(client)
    db.session.flush()

    # Créer bookings
    bookings = []
    for i in range(50):
        booking = Booking(
            customer_name=f"Patient {i}",
            pickup_location="Rue de Lausanne 1, Genève",
            dropoff_location="Hôpital Cantonal, Genève",
            scheduled_time=datetime.now() + timedelta(hours=i),
            amount=50.0,
            user_id=user.id,
            client_id=client.id,
            company_id=company.id,
            status=BookingStatus.PENDING
        )
        db.session.add(booking)
        bookings.append(booking)

    db.session.commit()
    return bookings
```

---

### Données de performance (load testing)

**Génération** :

```python
# session/new_files/profiling/generate_test_data.py
from faker import Faker
import json

fake = Faker('fr_CH')

data = {
    "users": [],
    "bookings": [],
    "drivers": []
}

for i in range(100):
    data["users"].append({
        "username": fake.user_name(),
        "email": fake.email(),
        "password": "Test123!",
        "role": "client"
    })

for i in range(500):
    data["bookings"].append({
        "customer_name": fake.name(),
        "pickup_location": fake.address(),
        "dropoff_location": fake.address(),
        "scheduled_time": fake.date_time_this_month().isoformat(),
        "amount": fake.random_int(30, 150)
    })

with open('test_data.json', 'w') as f:
    json.dump(data, f, indent=2)
```

---

## 📊 LOGS ATTENDUS

### Backend (succès)

```
INFO: Uvicorn running on http://0.0.0.0:5000
INFO: [INIT] Configuration Socket.IO...
INFO: ✅ Socket.IO initialisé: async_mode=eventlet
INFO: 🔧 [INIT] Enregistrement des routes et handlers Socket.IO...
INFO: ✅ Handlers Socket.IO chat enregistrés
INFO: 🔌 [CONNECT] HANDLER APPELÉ ! auth={'token': '...'}
INFO: 🧾 Token validé pour user test-user-123
INFO: ✅ Entreprise connectée à company_1
```

### Frontend (DevTools Console)

```
[CompanySocket] Connexion à: http://localhost:3000
✅ WebSocket connecté (company) xyz123
📡 Received driver_location_update: {driver_id: 5, latitude: 46.5, ...}
📡 Received dispatch_run_completed: {dispatch_run_id: 'abc', assignments_count: 12}
```

### Performance (wrk)

```
Running 30s test @ http://localhost:5000/api/bookings
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    78.52ms   45.23ms  450.12ms   89.23%
    Req/Sec   215.34     32.12   312.00     75.12%
  Latency Distribution
     50%   68ms
     75%   95ms
     90%  115ms
     99%  180ms
  25834 requests in 30.00s, 45.23MB read
Requests/sec:    861.13
Transfer/sec:      1.51MB
```

---

## 🆘 TROUBLESHOOTING

### Backend tests échouent

**Problème** : `sqlalchemy.exc.OperationalError: connection refused`

**Solution** :

```bash
# Vérifier PostgreSQL running
docker compose ps postgres
# Démarrer si nécessaire
docker compose up -d postgres
```

---

### Socket.IO connect handler pas appelé

**Problème** : Pas de log "🔌 [CONNECT] HANDLER APPELÉ !"

**Solution** :

```bash
# Vérifier async_mode
echo $SOCKETIO_ASYNC_MODE  # Doit être "eventlet" ou "gevent"

# Vérifier worker class Gunicorn
ps aux | grep gunicorn  # Doit contenir "--worker-class eventlet"

# Redémarrer avec bon worker
gunicorn wsgi:app --worker-class eventlet --bind 0.0.0.0:5000
```

---

### Frontend bundle toujours volumineux

**Problème** : Bundle size > 3 MB après patch

**Solution** :

```bash
# Vérifier que le patch est appliqué
grep "React.lazy" frontend/src/App.jsx  # Doit exister

# Purge node_modules + rebuild
rm -rf node_modules build
npm ci
npm run build

# Vérifier tree-shaking
npx webpack-bundle-analyzer build/bundle-stats.json
```

---

## 📝 CHECKLIST FINALE

Avant de marquer l'audit comme validé :

- [ ] ✅ Backend : Pytest 0 failed, coverage ≥75%
- [ ] ✅ Socket.IO : Connect fonctionne, événements reçus
- [ ] ✅ Frontend : Build OK, bundle <2.3MB, tests passent
- [ ] ✅ Mobile : Build EAS OK, tests Jest passent
- [ ] ✅ Performance : Latence p95 -20% min sur 3 endpoints
- [ ] ✅ Sécurité : Secrets .gitignore, JWT aud claim, PII scrubbing
- [ ] ✅ DB : Index créés, N+1 éliminés, migrations OK
- [ ] ✅ Dead files : Tous supprimés (15 fichiers)
- [ ] ✅ Linter : 0 error (Ruff + ESLint)
- [ ] ✅ Docker : Tous services healthy
- [ ] ✅ Monitoring : Logs structurés, métriques collectées

---

**Document validé par** : \***\*\_\*\***  
**Date** : \***\*\_\*\***
