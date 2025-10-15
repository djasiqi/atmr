# 📦 Guide d'Application des Patches ATMR

**Date**: 15 octobre 2025  
**Nombre total de patches**: 11

---

## 📋 Ordre d'Application Recommandé

### ✅ Phase 1: Correctifs Critiques Backend (Jour 1-2)

```bash
# 1. Timezone fixes (datetime.utcnow → datetime.now(timezone.utc))
git apply patches/backend_timezone_fix.patch
pytest backend/tests/  # Vérifier aucune régression

# 2. Celery config (acks_late, timeouts)
git apply patches/backend_celery_config.patch
# Redémarrer workers Celery pour appliquer config

# 3. Validation fixes (montants, enum, dead code)
git apply patches/backend_validation_fixes.patch
pytest backend/tests/test_models*.py

# 4. SocketIO validation stricte
git apply patches/backend_socketio_validation.patch
pytest backend/tests/test_socketio*.py
```

### ✅ Phase 2: Performance & Infra (Jour 2-3)

```bash
# 5. N+1 queries (joinedload)
git apply patches/backend_n+1_queries.patch
# Tests charge: vérifier /api/companies/me/bookings avec 1000+ bookings

# 6. PDF URLs config dynamique
git apply patches/backend_pdf_config.patch
# Ajouter PDF_BASE_URL dans backend/.env

# 7. Docker healthchecks
git apply patches/infra_docker_compose_healthchecks.patch
docker-compose up -d
docker-compose ps  # Vérifier tous healthy
```

### ✅ Phase 3: Migrations DB (Jour 3-4)

⚠️ **BACKUP DB OBLIGATOIRE AVANT**

```bash
# Backup production
pg_dump -h localhost -U atmr atmr > backup_pre_indexes_$(date +%Y%m%d).sql

# 8. Migration index critiques
cd backend
# Copier contenu backend_migration_indexes.patch dans nouvelle migration
alembic revision -m "add_critical_indexes"
# Éditer fichier généré, copier upgrade/downgrade depuis patch
alembic upgrade head

# Vérifier indexes créés
psql -U atmr atmr -c "\d+ booking"  # Voir tous les index
```

### ✅ Phase 4: Frontend (Jour 4-5)

```bash
# 9. JWT refresh automatique
cd frontend
git apply ../patches/frontend_jwt_refresh.patch
npm test

# 10. Setup tests (Jest + RTL + Cypress)
git apply ../patches/frontend_tests_setup.patch
git apply ../patches/frontend_e2e_cypress.patch

# Installer dépendances tests
npm install --save-dev @testing-library/react @testing-library/jest-dom \
  @testing-library/user-event @testing-library/react-hooks \
  axios-mock-adapter cypress

# Lancer tests
npm test
npx cypress run
```

### ✅ Phase 5: CI/CD & Tests Backend (Jour 5)

```bash
# 11. Backend tests complets
git apply patches/backend_tests_auth.patch

# Installer dépendances tests
cd backend
pip install pytest pytest-cov pytest-mock responses faker

# Lancer tests
pytest --cov=. --cov-report=html

# 12. CI/CD workflows
mkdir -p .github/workflows
cp ci/*.yml .github/workflows/

# Commit & push pour déclencher CI
git add .github/workflows/
git commit -m "ci: Add GitHub Actions workflows"
git push
```

### ✅ Phase 6: PII Logging (Optionnel GDPR)

```bash
# 13. PII masking dans logs
git apply patches/backend_pii_logging_fix.patch

# Activer dans .env
echo "MASK_PII_LOGS=true" >> backend/.env

# Tester logs masqués
tail -f backend/logs/app.log  # Vérifier emails → e***@***.com
```

---

## 🔍 Application Patch Individuel

### Méthode Standard (git apply)

```bash
# Vérifier patch avant application (dry-run)
git apply --check patches/backend_timezone_fix.patch

# Appliquer
git apply patches/backend_timezone_fix.patch

# Si conflit
git apply --3way patches/backend_timezone_fix.patch
# Résoudre conflits manuellement
```

### Méthode Alternative (patch command)

```bash
# GNU patch (si git apply échoue)
patch -p1 < patches/backend_timezone_fix.patch

# Dry-run
patch -p1 --dry-run < patches/backend_timezone_fix.patch
```

### Rollback Patch

```bash
# Inverser patch
git apply --reverse patches/backend_timezone_fix.patch

# Ou via patch
patch -p1 -R < patches/backend_timezone_fix.patch
```

---

## 🧪 Tests Post-Application

### Checklist Validation

Après **chaque** patch appliqué:

```bash
# Backend
cd backend
ruff check .  # Lint
pytest tests/ -v  # Tests (si existent)

# Frontend
cd frontend
npm run lint
npm test

# Infra
docker-compose config  # Valider YAML
docker-compose up -d
docker-compose ps  # Vérifier healthy
```

### Tests Smoke Complets

Après **tous** les patches:

```bash
# 1. Backend API
curl http://localhost:5000/health
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/companies/me

# 2. Frontend build
cd frontend
npm run build
ls -lh build/static/js/*.js  # Vérifier tailles

# 3. E2E critiques
npx cypress run --spec "cypress/e2e/company-flow.cy.js"

# 4. Celery tasks
docker-compose logs celery-worker  # Vérifier acks_late actif
```

---

## ⚠️ Conflits Potentiels

### Patch `backend_n+1_queries.patch`

**Conflit si**: Routes déjà modifiées localement

**Résolution**:
1. Appliquer manuellement les `joinedload()` sur routes conflictuelles
2. Comparer avec patch pour identifier imports manquants
3. Tester requêtes avant/après avec SQL EXPLAIN ANALYZE

### Patch `frontend_jwt_refresh.patch`

**Conflit si**: `apiClient.js` modifié pour autre raison

**Résolution**:
1. Fusionner manuellement logique refresh avec modifications locales
2. Garantir que `isRefreshing` flag et `failedQueue` sont bien ajoutés
3. Tester cycle complet: login → expiration token → refresh auto → requête OK

---

## 📊 Récapitulatif Patches

| Patch | Cible | Priorité | Effort | Risque |
|-------|-------|----------|--------|--------|
| `backend_timezone_fix.patch` | Services/Tasks | Critique | 5min | Faible |
| `backend_celery_config.patch` | Celery config | Critique | 2min | Faible |
| `backend_n+1_queries.patch` | Routes | Élevée | 3min | Moyen |
| `backend_pdf_config.patch` | PDF service | Élevée | 2min | Faible |
| `backend_validation_fixes.patch` | Models | Moyenne | 2min | Faible |
| `backend_socketio_validation.patch` | Sockets | Moyenne | 2min | Faible |
| `backend_pii_logging_fix.patch` | Logging | Moyenne | 5min | Faible |
| `backend_migration_indexes.patch` | DB schema | Critique | 10min | Moyen* |
| `backend_tests_auth.patch` | Tests | Élevée | 2min | Null |
| `frontend_jwt_refresh.patch` | Auth | Élevée | 3min | Moyen |
| `frontend_tests_setup.patch` | Tests | Moyenne | 2min | Null |
| `frontend_e2e_cypress.patch` | E2E | Moyenne | 5min | Null |
| `infra_docker_compose_healthchecks.patch` | Docker | Élevée | 2min | Faible |

\* Risque moyen si DB volumineuse (>100k rows), nécessite backup

**Total effort**: ~45 minutes (hors tests/validation)

---

## 🎯 Commande All-in-One (Dev/Staging)

```bash
#!/bin/bash
# apply_all_patches.sh - Appliquer tous les patches en une fois

set -e  # Exit on error

echo "🚀 Application de tous les patches ATMR..."

# Backend
git apply patches/backend_timezone_fix.patch
git apply patches/backend_celery_config.patch
git apply patches/backend_n+1_queries.patch
git apply patches/backend_pdf_config.patch
git apply patches/backend_validation_fixes.patch
git apply patches/backend_socketio_validation.patch
git apply patches/backend_pii_logging_fix.patch
git apply patches/backend_tests_auth.patch

# Frontend
git apply patches/frontend_jwt_refresh.patch
git apply patches/frontend_tests_setup.patch
git apply patches/frontend_e2e_cypress.patch

# Infra
git apply patches/infra_docker_compose_healthchecks.patch

# CI
mkdir -p .github/workflows
cp ci/*.yml .github/workflows/

echo "✅ Tous les patches appliqués avec succès"
echo "⚠️ N'oubliez pas d'appliquer les migrations DB manuellement"
echo "   cd backend && alembic upgrade head"
```

**Usage**:
```bash
chmod +x apply_all_patches.sh
./apply_all_patches.sh
```

⚠️ **ATTENTION PRODUCTION**: Appliquer patches un par un avec validation intermédiaire

---

## 📞 Support

En cas de problème:
1. Vérifier logs: `git apply` doit indiquer ligne(s) conflictuelle(s)
2. Appliquer manuellement section par section
3. Consulter REPORT.md pour contexte/justification
4. Tests régression: `pytest` backend, `npm test` frontend

---

*Guide généré le 15 octobre 2025. Patches testés sur arborescence du 15/10/2025.*

