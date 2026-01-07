# Dépendances Backend - ATMR

**Date :** 7 janvier 2025  
**Version :** 2.0.0 (Refactoring B1)

---

## 📋 Vue d'Ensemble

Ce document décrit les dépendances entre les modules du backend ATMR, incluant :
- Services internes
- Bounded contexts (DDD)
- Dépendances externes
- Module `unified_dispatch` v2.0

---

## 🗂️ Architecture Globale

```
backend/
├── bookings/          # Bounded Context: Gestion réservations (DDD)
├── drivers/           # Bounded Context: Gestion chauffeurs (DDD)
├── dispatch/          # Bounded Context: Dispatch & assignations (DDD)
├── companies/         # Bounded Context: Gestion entreprises (DDD)
├── shared/            # Code partagé (events, constants, utils)
├── infrastructure/    # Adapters, repositories, event bus
├── services/          # Services métier (dont unified_dispatch)
├── models/            # Models SQLAlchemy (legacy)
├── routes/            # Routes API (legacy + bounded contexts)
└── tasks/             # Tâches Celery
```

---

## 🎯 Module Principal: `unified_dispatch` v2.0

### Structure (Post-Refactoring B1)

```
services/unified_dispatch/
├── core/              # Types, exceptions, configuration fondamentale
├── data/              # Chargement et préparation données
├── optimization/      # Algorithmes OR-Tools, heuristiques
├── ml/                # Machine Learning & Reinforcement Learning
├── metrics/           # Métriques Prometheus, SLO, performance
├── validation/        # Contraintes métier, validation assignments
├── shadow_mode/       # A/B testing production
├── utils/             # Utilitaires (transactions, realtime, suggestions)
├── orchestration/     # Coordination pipeline dispatch
├── locking/           # Verrous distribués (Redis)
└── docs/              # Documentation module
```

### Dépendances Externes (unified_dispatch)

| Dépendance        | Version   | Usage                                      |
| ----------------- | --------- | ------------------------------------------ |
| **OR-Tools**      | >=9.5     | Résolution VRPTW (Vehicle Routing)         |
| **PyTorch**       | >=2.0     | Agent RL (DQN) pour scoring assignments   |
| **XGBoost**       | >=1.7     | Prédiction retards (ML)                    |
| **LightGBM**      | >=3.3     | Prédiction demande (ML)                    |
| **scikit-learn**  | >=1.3     | Preprocessing ML, clustering               |
| **Redis**         | >=6.2     | Locks distribués, cache, A/B tracking      |
| **PostgreSQL**    | >=14      | Stockage données (bookings, drivers, etc.) |
| **OSRM**          | >=5.27    | Matrices distance/durée (routing)          |
| **Prometheus**    | Client    | Métriques dispatch                         |

### Graphe de Dépendances Interne

```
orchestration → optimization, ml, metrics, validation, shadow_mode, utils, data
optimization → core, data, ml, metrics, validation
ml → core, data
data → core
validation → core
metrics → core
shadow_mode → core, metrics
utils → core
core → (aucune dépendance interne)
```

**Principe :** Dépendances unidirectionnelles (pas de cycles)

---

## 🧩 Bounded Contexts (DDD)

### 1. Bookings Context

**Responsabilité :** Gestion des réservations

**Structure :**
```
bookings/
├── domain/          # Entités, value objects, events
├── application/     # Use cases, services applicatifs
├── infrastructure/  # Repositories, adapters
└── presentation/    # Routes API
```

**Dépendances :**
- `shared.event_bus` (publish events)
- `infrastructure.repositories` (persistence)
- `models.Booking` (legacy, à migrer)

**Events émis :**
- `BookingCreatedEvent`
- `BookingUpdatedEvent`
- `BookingCancelledEvent`

---

### 2. Drivers Context

**Responsabilité :** Gestion des chauffeurs et disponibilités

**Structure :**
```
drivers/
├── domain/          # Entités, value objects, events
├── application/     # Use cases, services applicatifs
├── infrastructure/  # Repositories, adapters
└── presentation/    # Routes API
```

**Dépendances :**
- `shared.event_bus` (publish events)
- `infrastructure.repositories` (persistence)
- `models.Driver` (legacy, à migrer)

**Events émis :**
- `DriverCreatedEvent`
- `DriverUpdatedEvent`
- `DriverAvailabilityChangedEvent`

---

### 3. Dispatch Context

**Responsabilité :** Orchestration du dispatch et assignations

**Structure :**
```
dispatch/
├── domain/          # Entités, value objects, events
├── application/     # Use cases, services applicatifs
├── infrastructure/  # Adapters vers unified_dispatch
└── presentation/    # Routes API dispatch
```

**Dépendances :**
- `services.unified_dispatch` (via adapters)
- `shared.event_bus` (publish/subscribe events)
- `infrastructure.repositories` (persistence)
- `models.Assignment` (legacy, à migrer)

**Events émis :**
- `DriverNewBookingEvent` (via unified_dispatch)
- `AssignmentCreatedEvent`
- `AssignmentCancelledEvent`

**Events écoutés :**
- `BookingCreatedEvent` (trigger dispatch)
- `DriverAvailabilityChangedEvent` (re-dispatch)

---

### 4. Companies Context

**Responsabilité :** Gestion des entreprises et configurations

**Structure :**
```
companies/
├── domain/          # Entités, value objects, events
├── application/     # Use cases, services applicatifs
├── infrastructure/  # Repositories, adapters
└── presentation/    # Routes API
```

**Dépendances :**
- `shared.event_bus` (publish events)
- `infrastructure.repositories` (persistence)
- `models.Company` (legacy, à migrer)

**Events émis :**
- `CompanyCreatedEvent`
- `CompanyUpdatedEvent`
- `CompanyConfigChangedEvent`

---

## 🔄 Communication Inter-Contexts

### Event Bus (Architecture Hexagonale)

**Implémentation :**
- `shared.event_bus.EventBus` (interface)
- `infrastructure.event_bus.CeleryEventBus` (production)
- `infrastructure.event_bus.InMemoryEventBus` (tests)

**Pattern :** Publish/Subscribe asynchrone via Celery

**Exemples de flux :**

1. **Création Booking → Dispatch Automatique**
   ```
   Bookings Context
       └─> BookingCreatedEvent
           └─> Dispatch Context (listener)
               └─> Run Dispatch (unified_dispatch)
                   └─> DriverNewBookingEvent (notification)
   ```

2. **Changement Disponibilité Driver → Re-dispatch**
   ```
   Drivers Context
       └─> DriverAvailabilityChangedEvent
           └─> Dispatch Context (listener)
               └─> Trigger Re-dispatch
   ```

---

## 🛠️ Infrastructure & Shared

### `shared/`

**Contenu :**
- `constants.py` : Constantes globales (GeoConstants, etc.)
- `event_bus.py` : Interface EventBus
- `events.py` : Classes d'événements de domaine
- `time_utils.py` : Utilitaires temps (now_utc, etc.)
- `otel_setup.py` : Configuration OpenTelemetry

**Dépendances :** Aucune (module fondamental)

---

### `infrastructure/`

**Contenu :**
- `event_bus/` : Implémentations CeleryEventBus, InMemoryEventBus
- `repositories/` : Repositories SQLAlchemy pour bounded contexts
- `dispatch/` : Adapters vers `unified_dispatch` (validation, apply, etc.)

**Dépendances :**
- `shared.event_bus` (interface)
- `services.unified_dispatch` (pour adapters)
- `sqlalchemy`, `celery`, `redis`

---

## 📦 Dépendances Externes Globales

### Production (requirements.prod.txt)

| Catégorie            | Dépendance                   | Version  |
| -------------------- | ---------------------------- | -------- |
| **Web Framework**    | Flask                        | >=3.0    |
| **Database**         | SQLAlchemy                   | >=2.0    |
|                      | psycopg2-binary              | >=2.9    |
|                      | alembic                      | >=1.12   |
| **Task Queue**       | Celery                       | >=5.3    |
|                      | Redis                        | >=5.0    |
| **API**              | Flask-RESTX                  | >=1.2    |
|                      | marshmallow                  | >=3.20   |
| **Auth**             | Flask-JWT-Extended           | >=4.5    |
|                      | bcrypt                       | >=4.1    |
| **Optimisation**     | ortools                      | >=9.5    |
| **ML/RL**            | torch                        | >=2.0    |
|                      | xgboost                      | >=1.7    |
|                      | lightgbm                     | >=3.3    |
|                      | scikit-learn                 | >=1.3    |
|                      | numpy                        | >=1.24   |
|                      | pandas                       | >=2.0    |
| **Monitoring**       | prometheus-client            | >=0.18   |
|                      | sentry-sdk[flask]            | >=1.39   |
|                      | opentelemetry-*              | >=1.21   |
| **Security**         | cryptography                 | >=41.0   |
|                      | hvac                         | >=2.0    |
| **Utilities**        | requests                     | >=2.31   |
|                      | python-dotenv                | >=1.0    |
|                      | nplusone                     | >=1.0    |

### Développement (requirements.dev.txt)

| Catégorie            | Dépendance                   | Version  |
| -------------------- | ---------------------------- | -------- |
| **Testing**          | pytest                       | >=7.4    |
|                      | pytest-cov                   | >=4.1    |
|                      | pytest-flask                 | >=1.3    |
|                      | pytest-mock                  | >=3.12   |
| **Linting**          | ruff                         | >=0.1    |
|                      | basedpyright                 | >=1.32   |
| **Static Analysis**  | semgrep                      | >=1.50   |
| **Formatting**       | black                        | >=23.12  |
| **Type Checking**    | mypy                         | >=1.7    |

---

## 🔐 Services Externes

### OSRM (Routing)

**URL :** `http://osrm:5000` (Docker interne)  
**Usage :** Calcul matrices distance/durée pour dispatch  
**Fallback :** Cache Redis (si OSRM down)

### Vault (Secrets - Optionnel)

**URL :** Configurable via `VAULT_ADDR`  
**Usage :** Gestion secrets (DB passwords, API keys)  
**Fallback :** Variables d'environnement

### Sentry (Error Tracking)

**DSN :** Configurable via `SENTRY_DSN`  
**Usage :** Tracking erreurs + performance monitoring

### Prometheus (Métriques)

**Port :** `9090` (scraping)  
**Usage :** Collecte métriques dispatch, API, DB, Celery

---

## 📚 Documentation Complémentaire

- **Architecture `unified_dispatch`** : `docs/UNIFIED_DISPATCH_ARCHITECTURE.md`
- **Guide de migration v2.0** : `docs/UNIFIED_DISPATCH_MIGRATION_GUIDE.md`
- **Runbook** : `backend/RUNBOOK.md`
- **README principal** : `README.md`

---

## 🔄 Ordre de Démarrage (Dépendances Runtime)

1. **PostgreSQL** (port 5432)
2. **Redis** (port 6379)
3. **OSRM** (port 5000) - optionnel, cache disponible
4. **Backend API** (port 5000) - Flask
5. **Celery Worker** - consomme tasks
6. **Celery Beat** - schedule periodic tasks
7. **Prometheus** (port 9090) - scraping métriques
8. **Grafana** (port 3000) - dashboards

**Commande Docker Compose :**
```bash
docker-compose up -d postgres redis osrm
docker-compose up -d api celery-worker celery-beat
docker-compose up -d prometheus grafana
```

---

**Date de dernière mise à jour :** 7 janvier 2025  
**Version du document :** 2.0.0

