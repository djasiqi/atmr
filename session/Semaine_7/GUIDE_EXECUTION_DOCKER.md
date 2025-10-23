# 🐳 Guide d'Exécution - Semaine 7 (Docker + PostgreSQL)

**Date:** 20 octobre 2025  
**Environnement:** Docker + PostgreSQL

---

## 📋 Prérequis

- ✅ Docker et Docker Compose installés
- ✅ PostgreSQL configuré dans docker-compose.yml
- ✅ Services backend et database en cours d'exécution

---

## 🚀 Étape 1 : Démarrer les Services Docker

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier que les services sont lancés
docker-compose ps
```

**Services requis:**

- ✅ `backend` (Flask API)
- ✅ `db` ou `postgres` (PostgreSQL)
- ✅ `redis` (optionnel, pour Celery)

---

## 📊 Étape 2 : Exécuter la Migration Alembic

### Option A : Via Docker Compose (RECOMMANDÉ)

```bash
# Exécuter la migration dans le conteneur backend
docker-compose exec backend flask db upgrade

# Vérifier que la migration s'est bien passée
docker-compose exec backend flask db current
```

### Option B : Via Docker Run

```bash
# Si docker-compose exec ne fonctionne pas
docker exec -it atmr-backend-1 flask db upgrade
```

### Vérification de la migration

```bash
# Se connecter à PostgreSQL
docker-compose exec db psql -U postgres -d atmr_dev

# Dans psql, vérifier que la table existe
\dt autonomous_action

# Vérifier la structure
\d autonomous_action

# Quitter psql
\q
```

**Résultat attendu:**

```sql
Table "public.autonomous_action"
         Column          |            Type             | Modifiers
-------------------------+-----------------------------+-----------
 id                      | integer                     | not null
 company_id              | integer                     | not null
 booking_id              | integer                     |
 driver_id               | integer                     |
 action_type             | character varying(50)       | not null
 action_description      | character varying(500)      | not null
 action_data             | text                        |
 success                 | boolean                     | not null
 error_message           | text                        |
 execution_time_ms       | double precision            |
 confidence_score        | double precision            |
 expected_improvement_minutes | double precision       |
 trigger_source          | character varying(100)      |
 reviewed_by_admin       | boolean                     | not null
 reviewed_at             | timestamp without time zone |
 admin_notes             | text                        |
 created_at              | timestamp without time zone | not null
 updated_at              | timestamp without time zone | not null
```

---

## 🧪 Étape 3 : Exécuter les Tests

### Tests complets avec coverage

```bash
# Exécuter tous les tests de la Semaine 7
docker-compose exec backend pytest tests/test_safety_limits.py -v

# Avec rapport de couverture
docker-compose exec backend pytest tests/test_safety_limits.py \
  --cov=models.autonomous_action \
  --cov=services.unified_dispatch.autonomous_manager \
  --cov=routes.admin \
  --cov-report=html \
  --cov-report=term

# Voir le rapport HTML
# Le fichier sera dans backend/htmlcov/index.html
```

### Tests ciblés

```bash
# Tests du modèle uniquement
docker-compose exec backend pytest tests/test_safety_limits.py::TestAutonomousActionModel -v

# Tests du rate limiting uniquement
docker-compose exec backend pytest tests/test_safety_limits.py::TestSafetyLimits -v

# Tests d'intégration uniquement
docker-compose exec backend pytest tests/test_safety_limits.py::TestSafetyLimitsIntegration -v

# Tests du logging uniquement
docker-compose exec backend pytest tests/test_safety_limits.py::TestActionLogging -v
```

### Résultat attendu

```
tests/test_safety_limits.py::TestAutonomousActionModel::test_create_autonomous_action PASSED
tests/test_safety_limits.py::TestAutonomousActionModel::test_count_actions_last_hour PASSED
tests/test_safety_limits.py::TestAutonomousActionModel::test_count_actions_today PASSED
tests/test_safety_limits.py::TestAutonomousActionModel::test_count_actions_by_type PASSED
tests/test_safety_limits.py::TestAutonomousActionModel::test_to_dict PASSED
tests/test_safety_limits.py::TestSafetyLimits::test_check_safety_limits_ok PASSED
tests/test_safety_limits.py::TestSafetyLimits::test_hourly_limit_reached PASSED
tests/test_safety_limits.py::TestSafetyLimits::test_daily_limit_reached PASSED
tests/test_safety_limits.py::TestSafetyLimits::test_action_type_hourly_limit PASSED
tests/test_safety_limits.py::TestSafetyLimits::test_action_type_daily_limit PASSED
tests/test_safety_limits.py::TestSafetyLimits::test_failed_actions_not_counted PASSED
tests/test_safety_limits.py::TestSafetyLimits::test_different_companies_isolated PASSED
tests/test_safety_limits.py::TestSafetyLimitsIntegration::test_multiple_action_types_independent_limits PASSED
tests/test_safety_limits.py::TestSafetyLimitsIntegration::test_hourly_limit_resets_after_hour PASSED
tests/test_safety_limits.py::TestSafetyLimitsIntegration::test_action_logging_updates_limits PASSED
tests/test_safety_limits.py::TestActionLogging::test_successful_action_logged PASSED
tests/test_safety_limits.py::TestActionLogging::test_failed_action_logged PASSED
tests/test_safety_limits.py::TestActionLogging::test_action_blocked_by_limits_not_logged PASSED

==================== 19 passed in 2.34s ====================
```

---

## 🔍 Étape 4 : Tester les Endpoints Admin

### 4.1 Démarrer le serveur (si pas déjà fait)

```bash
# Vérifier que le backend est lancé
docker-compose logs -f backend
```

### 4.2 Tester avec curl ou Postman

#### Obtenir un token JWT (admin)

```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "votre_mot_de_passe"
  }'
```

**Sauvegarder le token retourné:**

```bash
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

#### Tester l'endpoint de liste

```bash
# Liste des actions autonomes
curl -X GET "http://localhost:5000/api/admin/autonomous-actions?page=1&per_page=20" \
  -H "Authorization: Bearer $TOKEN"
```

#### Tester l'endpoint de statistiques

```bash
# Stats des dernières 24h
curl -X GET "http://localhost:5000/api/admin/autonomous-actions/stats?period=day" \
  -H "Authorization: Bearer $TOKEN"
```

#### Tester l'endpoint de détail

```bash
# Détails d'une action spécifique (remplacer 1 par un ID existant)
curl -X GET "http://localhost:5000/api/admin/autonomous-actions/1" \
  -H "Authorization: Bearer $TOKEN"
```

#### Tester l'endpoint de review

```bash
# Marquer une action comme reviewée
curl -X POST "http://localhost:5000/api/admin/autonomous-actions/1/review" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "notes": "Action validée - Tout est OK"
  }'
```

---

## 📊 Étape 5 : Vérifier les Données dans PostgreSQL

### Se connecter à la base

```bash
# Via docker-compose
docker-compose exec db psql -U postgres -d atmr_dev

# Ou directement
docker exec -it atmr-db-1 psql -U postgres -d atmr_dev
```

### Requêtes utiles

```sql
-- Compter le nombre d'actions
SELECT COUNT(*) FROM autonomous_action;

-- Actions par type
SELECT action_type, COUNT(*),
       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful
FROM autonomous_action
GROUP BY action_type;

-- Actions de la dernière heure
SELECT * FROM autonomous_action
WHERE created_at >= NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;

-- Actions par entreprise
SELECT company_id, COUNT(*),
       AVG(execution_time_ms) as avg_time
FROM autonomous_action
GROUP BY company_id;

-- Actions non reviewées
SELECT COUNT(*) FROM autonomous_action
WHERE reviewed_by_admin = false;

-- Top 10 actions les plus récentes
SELECT id, action_type, success, created_at, action_description
FROM autonomous_action
ORDER BY created_at DESC
LIMIT 10;
```

---

## 🐛 Dépannage

### Problème : Migration ne s'exécute pas

```bash
# Vérifier l'état des migrations
docker-compose exec backend flask db current

# Voir l'historique
docker-compose exec backend flask db history

# Forcer l'upgrade
docker-compose exec backend flask db upgrade head
```

### Problème : Table déjà existante

```bash
# Si la table existe déjà, marquer la migration comme appliquée
docker-compose exec backend flask db stamp abc123456789
```

### Problème : Tests échouent

```bash
# Nettoyer la base de test
docker-compose exec backend pytest tests/test_safety_limits.py --create-db

# Réinitialiser les fixtures
docker-compose exec backend pytest --cache-clear
```

### Problème : Imports ne fonctionnent pas

```bash
# Vérifier que le modèle est bien importé
docker-compose exec backend python -c "from models.autonomous_action import AutonomousAction; print('OK')"

# Vérifier l'import dans __init__.py
docker-compose exec backend python -c "from models import AutonomousAction; print('OK')"
```

### Problème : Connexion PostgreSQL

```bash
# Vérifier les variables d'environnement
docker-compose exec backend env | grep SQLALCHEMY

# Tester la connexion
docker-compose exec backend python -c "from db import db; from app import create_app; app = create_app(); print('DB OK')"
```

---

## 📝 Logs et Monitoring

### Voir les logs en temps réel

```bash
# Logs du backend
docker-compose logs -f backend

# Logs PostgreSQL
docker-compose logs -f db

# Tous les logs
docker-compose logs -f
```

### Filtrer les logs des actions autonomes

```bash
# Rechercher "AutonomousManager" dans les logs
docker-compose logs backend | grep "AutonomousManager"

# Rechercher les actions bloquées
docker-compose logs backend | grep "blocked_by_limits"

# Rechercher les limites atteintes
docker-compose logs backend | grep "Limite.*atteinte"
```

---

## ✅ Checklist de Validation

- [ ] Services Docker démarrés
- [ ] Migration Alembic exécutée
- [ ] Table `autonomous_action` créée dans PostgreSQL
- [ ] 7 index créés sur la table
- [ ] 19 tests passent (100% success)
- [ ] Couverture de tests >= 90%
- [ ] 4 endpoints admin accessibles
- [ ] Authentification JWT fonctionne
- [ ] Rate limiting opérationnel
- [ ] Logging automatique fonctionne
- [ ] Requêtes PostgreSQL fonctionnent
- [ ] Pas d'erreurs dans les logs

---

## 🚀 Commandes Rapides

### Setup complet (première fois)

```bash
# Tout en une fois
docker-compose up -d && \
docker-compose exec backend flask db upgrade && \
docker-compose exec backend pytest tests/test_safety_limits.py -v
```

### Vérification rapide

```bash
# Vérifier que tout fonctionne
docker-compose ps && \
docker-compose exec db psql -U postgres -d atmr_dev -c "\dt autonomous_action" && \
docker-compose exec backend pytest tests/test_safety_limits.py --co
```

### Reset complet (si problème)

```bash
# ATTENTION : Supprime toutes les données !
docker-compose down -v && \
docker-compose up -d && \
docker-compose exec backend flask db upgrade && \
docker-compose exec backend pytest tests/test_safety_limits.py -v
```

---

## 📚 Documentation Supplémentaire

- **README principal:** `session/Semaine_7/SEMAINE_7_SAFETY_AUDIT_TRAIL_COMPLETE.md`
- **Migration:** `backend/migrations/versions/abc123456789_add_autonomous_action_table.py`
- **Modèle:** `backend/models/autonomous_action.py`
- **Tests:** `backend/tests/test_safety_limits.py`
- **Routes Admin:** `backend/routes/admin.py` (lignes 342-620)

---

## 🎯 Prochaines Étapes

Une fois que tout est validé ✅, vous pouvez passer à :

1. **Semaine 9 : Intégration ML Pipeline**

   - Feature flags configuration
   - Endpoint admin ML
   - Intégration dans engine.py

2. **Tests en environnement de production**
   - Monitoring des métriques
   - Ajustement des limites de sécurité
   - Validation avec données réelles

---

_Guide d'exécution Docker + PostgreSQL_  
_Semaine 7 : Safety & Audit Trail_  
_Généré le 20 octobre 2025_
