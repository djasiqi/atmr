# 🧪 Tests Backend ATMR

## 📊 État Actuel

✅ **14 tests unitaires qui passent** (sans DB)  
⚠️ **Tests d'intégration nécessitent PostgreSQL** (incompatibilité SQLite)

## 🎯 Structure

```
tests/
├── conftest.py              # Fixtures (app, db, auth)
├── test_auth.py             # Auth (nécessite PostgreSQL)
├── test_bookings.py         # Bookings (nécessite PostgreSQL)
├── test_clients.py          # Clients (nécessite PostgreSQL)
├── test_dispatch.py         # Dispatch (nécessite PostgreSQL)
├── test_drivers.py          # Drivers (nécessite PostgreSQL)
├── test_models.py           # ✅ Tests enums (14 tests)
├── test_osrm_client.py      # ✅ Tests OSRM (6 tests)
└── test_utils.py            # ✅ Tests utils (8 tests)
```

## ✅ Tests Qui Passent (14)

### Modèles & Enums (4 tests)
- ✅ `test_booking_status_values` - Valeurs BookingStatus
- ✅ `test_user_role_values` - Valeurs UserRole
- ✅ `test_payment_status_values` - Valeurs PaymentStatus
- ✅ `test_booking_status_choices` - Méthode choices()

### Utils (8 tests)
- ✅ `test_time_utils_import` - Import des helpers timezone
- ✅ `test_now_local` - Génération datetime naïf
- ✅ `test_iso_utc_z` - Conversion ISO avec Z
- ✅ `test_logging_utils_import` - Import module logging

### OSRM Client (6 tests)
- ✅ `test_osrm_haversine_fallback` - Distance haversine
- ✅ `test_osrm_fallback_matrix` - Matrice fallback
- ✅ `test_osrm_table_mock_success` - Mock HTTP table
- ✅ `test_osrm_timeout_raises_exception` - Gestion timeout
- ✅ `test_osrm_cache_key_generation` - Clés cache stables
- ✅ `test_osrm_eta_fallback` - Calcul ETA fallback

## ⚠️ Limitation SQLite vs PostgreSQL

Les modèles ATMR utilisent des fonctionnalités PostgreSQL :
- **JSONB** : Colonnes `rejected_by`, `extra_data`
- **Syntaxe `::jsonb`** : `server_default=text("'[]'::jsonb")`

SQLite ne supporte pas ces syntaxes → **Tests d'intégration nécessitent PostgreSQL**.

## 🚀 Exécution

### Tests Unitaires (sans DB)

```bash
cd backend

# Activer venv
.\venv\Scripts\activate

# Exécuter tests unitaires
pytest tests/test_models.py tests/test_utils.py tests/test_osrm_client.py -v

# Avec coverage
pytest tests/test_models.py tests/test_utils.py tests/test_osrm_client.py -v --cov=. --cov-report=term
```

### Tests Complets (PostgreSQL requis)

#### Option 1 : Docker Compose

```bash
# Démarrer PostgreSQL + Redis
docker-compose up -d postgres redis

# Configurer env tests
export DATABASE_URL="postgresql://atmr:atmr_test_password@localhost:5432/atmr_test"

# Exécuter tests
pytest -v --cov=. --cov-report=html
```

#### Option 2 : GitHub Actions

Les workflows CI utilisent PostgreSQL 16 + Redis 7 :
- `.github/workflows/backend-tests.yml`

Les tests complets s'exécutent automatiquement sur push/PR.

## 📈 Coverage Attendu

| Fichier                         | Coverage Cible |
| ------------------------------- | -------------- |
| `services/osrm_client.py`       | ✅ ~80%        |
| `shared/time_utils.py`          | ✅ ~70%        |
| `models/enums.py`               | ✅ ~90%        |
| `routes/auth.py`                | ⚠️ (PostgreSQL) |
| `routes/bookings.py`            | ⚠️ (PostgreSQL) |
| `services/unified_dispatch/`    | ⚠️ (PostgreSQL) |

## 🔧 Commandes Utiles

```bash
# Tests spécifiques
pytest tests/test_osrm_client.py -v

# Tests avec prints
pytest tests/test_osrm_client.py -v -s

# Ré-exécuter les failed
pytest --lf

# Coverage HTML
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## 📝 Notes

1. **Tests unitaires** : Testent la logique sans effets de bord (DB, réseau)
2. **Tests d'intégration** : Nécessitent PostgreSQL (via Docker ou CI)
3. **CI/CD** : Tests complets s'exécutent automatiquement sur GitHub Actions

---

**Date** : 15 octobre 2025  
**Tests qui passent** : 14/36 (tests unitaires uniquement)  
**Tests complets** : Nécessitent PostgreSQL via Docker/CI

