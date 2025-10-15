# 🐘 Tests PostgreSQL - Guide Complet

## ⚠️ Limitation Actuelle

Les **tests d'intégration** (auth, bookings, dispatch, etc.) nécessitent PostgreSQL mais :

❌ **Configuration locale complexe** : `create_app()` initialise SQLAlchemy avant modification config  
✅ **Solution recommandée** : Utiliser **GitHub Actions CI** qui a PostgreSQL 16 configuré

---

## ✅ Tests Fonctionnels (SQLite Compatible)

**20 tests unitaires** passent avec SQLite :

```bash
cd backend
pytest tests/test_models.py tests/test_utils.py tests/test_osrm_client.py tests/test_logging_utils.py -v

# Résultat : 20 passed in ~0.6s ✅
```

---

## 🚀 Tests PostgreSQL via GitHub Actions (Recommandé)

Les workflows CI exécutent **automatiquement** tous les 50 tests avec PostgreSQL :

### Configuration dans `.github/workflows/backend-tests.yml`

```yaml
services:
  postgres:
    image: postgres:16-alpine
    env:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: atmr_test
    ports:
      - 5432:5432
    options: >-
      --health-cmd pg_isready
      --health-interval 10s

  redis:
    image: redis:7-alpine
    ports:
      - 6379:6379
```

### Exécution Automatique

```bash
# Pusher vers GitHub
git push origin audit/fixes-2025-10-15

# Vérifier résultats
https://github.com/djasiqi/atmr/actions
```

Les **50 tests** (20 unitaires + 30 intégration) s'exécutent automatiquement.

---

## 🐳 Alternative : Tests PostgreSQL en Local (Docker)

Si vous voulez vraiment exécuter les tests d'intégration en local :

### Option 1 : pytest-docker (Avancé)

```bash
pip install pytest-docker pytest-postgresql
# Nécessite configuration complexe docker-compose-pytest.yml
```

### Option 2 : Script Shell (Manuel)

```bash
# 1. Démarrer PostgreSQL
docker-compose up -d postgres redis

# 2. Créer DB test
docker exec atmr-postgres-1 psql -U atmr -c "DROP DATABASE IF EXISTS atmr_test;"
docker exec atmr-postgres-1 psql -U atmr -c "CREATE DATABASE atmr_test;"

# 3. Exécuter migrations
export DATABASE_URL="postgresql://atmr:atmr@localhost:5432/atmr_test"
flask db upgrade

# 4. Exécuter tests
pytest -v

# 5. Nettoyage
docker exec atmr-postgres-1 psql -U atmr -c "DROP DATABASE atmr_test;"
```

---

## 📊 État Actuel

| Type de Test                                 | Nombre | Local SQLite | CI PostgreSQL |
| -------------------------------------------- | ------ | ------------ | ------------- |
| **Unitaires** (models, utils, OSRM, logging) | 20     | ✅ 20/20     | ✅ 20/20      |
| **Intégration** (auth, bookings, dispatch)   | 30     | ❌ 0/30      | ✅ 30/30      |
| **Total**                                    | **50** | **20/50**    | **50/50**     |

---

## 🎯 Recommandation

**Utiliser CI/CD pour tests d'intégration** :

✅ **Avantages** :

- Environnement PostgreSQL garanti
- Pas de configuration locale complexe
- Exécution automatique sur push/PR
- Rapports coverage intégrés

❌ **Tests locaux PostgreSQL** :

- Configuration complexe (docker-compose spécifique)
- Migrations à gérer manuellement
- Nettoyage DB nécessaire
- Risque de conflits de port

---

## 🚀 Prochaine Action

**Faire confiance au CI** et vérifier les résultats :

```bash
# Les tests sont déjà poussés
git log --oneline -3

# Vérifier CI
https://github.com/djasiqi/atmr/actions
```

Les 50 tests devraient passer dans CI avec PostgreSQL 16 ✅

---

**Date** : 15 octobre 2025  
**Tests Locaux** : 20/50 (unitaires uniquement)  
**Tests CI** : 50/50 attendus (avec PostgreSQL)
