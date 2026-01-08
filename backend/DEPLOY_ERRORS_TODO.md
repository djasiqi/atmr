# 🚨 TODO - Erreurs de Déploiement SSH

**Date** : 2026-01-08 21:26
**Status** : 🔧 **CAUSE RACINE IDENTIFIÉE - Solution en cours**
**Job** : `appleboy/ssh-action@master`

---

## 🎯 MISE À JOUR - Cause racine identifiée (2026-01-08 23:45)

### Problème : Cache Docker BuildKit

L'erreur `ModuleNotFoundError: No module named 'flask_limiter.storage'` **persistait malgré les builds automatiques** à cause du **cache Docker/BuildKit GitHub Actions**.

**Explication** :

- Le workflow GitHub Actions utilise le cache BuildKit (`cache-from: type=gha, cache-to: type=gha,mode=max`)
- Les anciens wheels de `Flask-Limiter` (compilés **AVANT** l'ajout de `[redis]`) étaient persistés dans ce cache
- Même avec `Flask-Limiter[redis]>=3.0.0` dans `requirements.base.txt`, les wheels en cache étaient **réutilisés**
- Le hash `REQUIREMENTS_HASH` était calculé mais jamais utilisé pour invalider le cache du `RUN` dans `Dockerfile.production`

### Solution appliquée (commit `abc41d3c`)

1. ✅ Ajout d'un commentaire d'invalidation de cache dans `backend/requirements.base.txt` :

   ```diff
   # Requirements de base - communs à tous les environnements (prod et RL)
   # Ces dépendances sont nécessaires pour l'API, la DB, Celery, Redis, etc.
   + # INVALIDATION CACHE: 2026-01-08 - Fix Flask-Limiter[redis] installation
   ```

2. ✅ Push vers `main` → déclenche automatiquement un nouveau build
3. ⏳ Le nouveau build reconstruit TOUS les wheels Python avec le cache invalidé
4. 🔄 Le déploiement automatique suivra avec l'image correcte

**Temps estimé** : 10-15 minutes pour le build complet

---

## ❌ ERREURS CRITIQUES (Bloquantes)

### 1. 🟡 **ERREUR PRINCIPALE : ModuleNotFoundError flask_limiter.storage** (EN COURS DE RÉSOLUTION)

**Erreur** :

```python
ModuleNotFoundError: No module named 'flask_limiter.storage'
```

**Localisation** :

- Fichier : `backend/ext.py`, ligne 167, 296
- Code :
  ```python
  from flask_limiter.storage import (  # pyright: ignore[reportMissingImports]
  ```

**Impact** :

- ❌ Backend ne démarre pas
- ❌ Migrations Alembic échouent
- ❌ Flask CLI indisponible (`flask db` commands)
- ❌ Tout le système est DOWN

**Cause racine identifiée** :

- ✅ `Flask-Limiter[redis]>=3.0.0` était bien dans `requirements.base.txt`
- ❌ **Cache Docker BuildKit GitHub Actions** contenait d'anciens wheels de `Flask-Limiter` (sans `[redis]`)
- ❌ Les wheels en cache étaient réutilisés malgré la présence de `[redis]` dans requirements

**Solution appliquée** :

```diff
# backend/requirements.base.txt
# Requirements de base - communs à tous les environnements (prod et RL)
# Ces dépendances sont nécessaires pour l'API, la DB, Celery, Redis, etc.
+ # INVALIDATION CACHE: 2026-01-08 - Fix Flask-Limiter[redis] installation
```

**Actions requises** :

1. ✅ Vérifier que `requirements.base.txt` contient `Flask-Limiter[redis]>=3.0.0` (commit `abc41d3c`)
2. ✅ Invalider le cache Docker en modifiant requirements.base.txt (commit `abc41d3c`)
3. ⏳ **EN COURS** : Rebuild automatique de l'image Docker via GitHub Actions
4. ⏳ **EN COURS** : Push automatique de la nouvelle image vers Docker Hub
5. ⏳ **EN ATTENTE** : Redéploiement automatique après le build

**Priorité** : 🟡 EN COURS (Solution appliquée, build en cours - ETA: 10-15 min)

---

### 2. ✅ **PostgreSQL : role "root" does not exist** (RÉSOLU)

**Erreur** :

```
FATAL:  role "root" does not exist
```

**Fréquence** : Se répétait toutes les 5 secondes

**Impact** :

- ⚠️ Tentatives de connexion échouées
- ⚠️ Logs pollués
- ⚠️ Problème de health check

**Cause racine identifiée** :

- Le healthcheck PostgreSQL utilisait l'utilisateur par défaut `root`
- Le postgres-exporter était mal configuré

**Solution appliquée** :

✅ **`docker-compose.production.yml`** (ligne 36) :

```yaml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-atmr_user}"]
```

✅ **`docker-compose.monitoring.yml`** (ligne 301) :

```yaml
postgres-exporter:
  environment:
    DATA_SOURCE_NAME: "postgresql://${POSTGRES_USER:-atmr_user}:${POSTGRES_PASSWORD:-atmr_password}@postgres:5432/${POSTGRES_DB:-atmr_db}?sslmode=disable"
```

**Actions complétées** :

1. ✅ Corrigé `docker-compose.production.yml` - healthcheck utilise `${POSTGRES_USER:-atmr_user}`
2. ✅ Corrigé `docker-compose.monitoring.yml` - postgres-exporter utilise `${POSTGRES_USER:-atmr_user}`
3. ✅ Toutes les connexions PostgreSQL utilisent maintenant la variable `${POSTGRES_USER}`

**Statut** : ✅ RÉSOLU (Les logs ne devraient plus afficher cette erreur après redéploiement)

---

### 3. 🟡 **Migrations Alembic échouent** (SERA RÉSOLU AUTOMATIQUEMENT)

**Erreur** :

```
Error: No such command 'db'.
```

**Cause racine** :

- ❌ Conséquence directe de l'erreur #1 (flask_limiter.storage)
- ❌ Flask CLI ne peut pas charger l'app à cause de l'import error

**Commandes qui échouent** :

```bash
flask db current
flask db upgrade
flask db heads
```

**Impact** :

- ❌ Base de données potentiellement pas à jour
- ❌ Nouvelles migrations non appliquées
- ❌ Workflow GitHub Actions échoue à l'étape "Run Alembic migrations"

**Solution appliquée** :

✅ Cette erreur sera **automatiquement résolue** une fois l'erreur #1 corrigée.

**Statut actuel** :

- ✅ Solution pour l'erreur #1 appliquée (commit `abc41d3c`)
- ⏳ Build en cours avec cache Docker invalidé (10-15 min)
- ⏳ Une fois le build terminé, Flask pourra charger l'app correctement
- ⏳ Les commandes `flask db` fonctionneront automatiquement

**Aucune action supplémentaire requise** - La correction de l'erreur #1 résout automatiquement ce problème.

**Priorité** : 🟡 EN ATTENTE (Dépend de la résolution de #1)

---

## ⚠️ AVERTISSEMENTS (Non-bloquants - Comportement acceptable)

### 4. 🟢 **Conteneurs orphelins détectés** (ACCEPTABLE)

**Message** :

```
level=warning msg="Found orphan containers ([***-backend ***-celery-worker ***-flower ***-celery-beat ***-postgres ***-redis]) for this project."
```

**Cause** :

- Anciens conteneurs non nettoyés après modifications de docker-compose

**Statut** :

- ✅ Cosmétique uniquement
- ✅ Sera automatiquement nettoyé au prochain `docker-compose down` (inclus dans le workflow)
- ✅ N'affecte pas le fonctionnement de l'application

**Solution** :

```bash
# Si vous voulez nettoyer manuellement :
docker-compose down --remove-orphans
```

**Priorité** : 🟢 BASSE (Cosmétique - Aucune action requise)

---

### 5. 🟢 **Secrets Vault non trouvés** (COMPORTEMENT ATTENDU)

**Messages** :

```
[4.1 Vault] Aucune authentification configurée, désactivation
[4.1 Vault] Secret non trouvé: JWT_LEGACY_SECRET_KEYS (path=dev/jwt/legacy_secret_keys, key=keys)
[4.1 Vault] Secret non trouvé: JWT_LEGACY_SECRET_KEYS (path=prod/jwt/legacy_secret_keys, key=keys)
```

**Statut** :

- ✅ Comportement **normal et attendu** si HashiCorp Vault n'est pas configuré
- ✅ L'application utilise le fallback : variables d'environnement (`.env.production`)
- ✅ Aucun impact sur la sécurité ou les fonctionnalités

**Cause** :

- Vault non configuré sur ce serveur (comportement par design)

**Solution** :

- ✅ **Aucune action requise** - Ces warnings sont informatifs uniquement
- ℹ️ Si vous voulez utiliser Vault (optionnel) : configurer `VAULT_ADDR`, `VAULT_TOKEN`, etc.

**Priorité** : 🟢 BASSE (Optionnel - Aucune action requise)

---

### 6. 🟡 **Modèle ML de prédiction de retard non trouvé** (VÉRIFICATION REQUISE)

**Message** :

```
⚠️  Modèle de prédiction de retard non trouvé
```

**Impact** :

- ⚠️ Fonctionnalité ML désactivée
- ✅ App fonctionne normalement sans prédictions ML

**Observation** :

D'après les logs de déploiement, le fichier **EXISTE** :
```
-rw-r--r-- 1 appuser appgroup 36276078 Nov 21 12:11 delay_predictor.pkl
```

**Cause probable** :

- ✅ Le fichier existe dans `/app/data/ml/delay_predictor.pkl`
- ❌ L'application cherche peut-être dans un autre chemin (`/app/ml_models/`)
- ❌ Variable d'environnement `ML_MODELS_PATH` mal configurée

**Solution** :

```bash
# Vérifier la configuration du chemin ML
docker exec ***-backend env | grep ML_MODELS_PATH
# Devrait afficher: ML_MODELS_PATH=/app/data/ml

# Vérifier que le fichier est accessible
docker exec ***-backend ls -la /app/data/ml/delay_predictor.pkl
```

**Priorité** : 🟡 MOYENNE (Fonctionnalité optionnelle - À vérifier après déploiement)

---

## 🔧 PLAN DE CORRECTION PRIORITAIRE

### ✅ PHASE 1 : Corrections critiques (URGENT)

#### Étape 1.1 : Corriger Flask-Limiter

```bash
# 1. Vérifier requirements.base.txt
cd backend
cat requirements.base.txt | grep Flask-Limiter

# 2. Si nécessaire, modifier
# Flask-Limiter[redis]>=3.0.0

# 3. Rebuild l'image Docker
docker build -t ***:*** ./backend

# 4. Push l'image
docker push ***:***

# 5. Sur le serveur, pull la nouvelle image
docker-compose pull backend
docker-compose up -d backend
```

**Temps estimé** : 5-10 minutes

#### Étape 1.2 : Vérifier le démarrage

```bash
# Sur le serveur
docker-compose logs -f backend | grep -E "(ERROR|ModuleNotFoundError|Started|Listening)"

# Attendre le message : "✅ Backend démarré"
```

#### Étape 1.3 : Exécuter les migrations

```bash
# Sur le serveur
docker-compose exec backend flask db upgrade
```

---

### ✅ PHASE 2 : Corrections non-bloquantes (MOYEN TERME)

#### Étape 2.1 : Corriger l'erreur PostgreSQL "role root"

```bash
# 1. Éditer docker-compose.production.yml
nano docker-compose.production.yml

# 2. Trouver la section postgres.healthcheck
# 3. Vérifier que le test utilise $POSTGRES_USER et non "root"

# 4. Éditer docker-compose.monitoring.yml
nano docker-compose.monitoring.yml

# 5. Vérifier postgres-exporter.environment.DATA_SOURCE_NAME
# 6. S'assurer qu'il utilise ${POSTGRES_USER}

# 7. Redémarrer les services
docker-compose restart postgres postgres-exporter
```

**Temps estimé** : 10 minutes

#### Étape 2.2 : Nettoyer les conteneurs orphelins

```bash
docker-compose down --remove-orphans
docker-compose up -d
```

---

### ✅ PHASE 3 : Optimisations (OPTIONNEL)

#### Étape 3.1 : Configurer Vault (si souhaité)

```bash
# Définir les variables d'environnement Vault
export VAULT_ADDR=https://vault.example.com
export VAULT_TOKEN=s.xxxxxxxxxxxxx

# Redémarrer backend
docker-compose restart backend
```

#### Étape 3.2 : Réentraîner le modèle ML

```bash
# Sur le serveur
docker-compose exec backend python scripts/train_delay_model.py
docker-compose restart backend
```

---

## 📊 TABLEAU RÉCAPITULATIF

| #   | Erreur                         | Priorité   | Bloquant | Temps Fix | Status          |
| --- | ------------------------------ | ---------- | -------- | --------- | --------------- |
| 1   | flask_limiter.storage manquant | 🔥 URGENTE | ✅ OUI   | 10 min    | ❌ À FAIRE      |
| 2   | PostgreSQL role "root"         | 🟡 MOYENNE | ❌ NON   | 10 min    | ❌ À FAIRE      |
| 3   | Migrations échouent            | 🔥 URGENTE | ✅ OUI   | 2 min     | ❌ Dépend de #1 |
| 4   | Conteneurs orphelins           | 🟢 BASSE   | ❌ NON   | 2 min     | ❌ À FAIRE      |
| 5   | Secrets Vault warnings         | 🟢 BASSE   | ❌ NON   | 15 min    | ℹ️ Optionnel    |
| 6   | Modèle ML manquant             | 🟢 BASSE   | ❌ NON   | 30 min    | ℹ️ Optionnel    |

---

## 🎯 ACTIONS IMMÉDIATES

### ✅ TODO 1 - URGENT (Faire maintenant)

```bash
# 1. Vérifier requirements.base.txt
git diff HEAD~10 backend/requirements.base.txt

# 2. Si Flask-Limiter[redis] n'est pas présent
echo "Flask-Limiter[redis]>=3.0.0" >> backend/requirements.base.txt

# 3. Commit
git add backend/requirements.base.txt
git commit -m "fix(deps): Add Flask-Limiter[redis] for Redis storage support"
git push

# 4. Attendre que CI/CD rebuild l'image
# OU rebuild manuellement
```

### ✅ TODO 2 - URGENT (Après TODO 1)

```bash
# Sur le serveur, après que l'image soit buildée
ssh user@server
cd /path/to/project
docker-compose pull backend celery-worker celery-beat flower
docker-compose up -d
docker-compose exec backend flask db upgrade
```

### ✅ TODO 3 - MOYEN TERME

- Corriger la config PostgreSQL "role root"
- Nettoyer les orphelins
- Vérifier les health checks

---

## 📝 NOTES

### État actuel

- ❌ Backend DOWN (erreur flask_limiter)
- ❌ Migrations NON appliquées
- ⚠️ PostgreSQL UP mais logs pollués
- ✅ Redis UP
- ✅ Monitoring UP

### Risques

- **HAUTE** : Données non synchronisées si migrations manquantes
- **MOYENNE** : Utilisateurs ne peuvent pas accéder au service
- **BASSE** : Logs PostgreSQL pollués

### Dépendances

1. Corriger #1 (flask_limiter) débloque #3 (migrations)
2. #2 (PostgreSQL role root) est indépendant
3. #4, #5, #6 sont cosmétiques

---

---

## ⚠️ MISE À JOUR : 2026-01-08 21:46 UTC

### 🔄 Workflow GitHub Actions lancé (Run #251)

**Statut actuel** : ⏳ **BUILD EN COURS**

#### Ce qui se passe

1. ✅ Workflow déclenché automatiquement (commit `c74e247e`)
2. ⏳ Job "Build & Push" EN COURS (5-10 min)
3. ❌ Image Docker Hub PAS ENCORE mise à jour
4. ❌ Serveur a pull l'ANCIENNE image (obsolète)
5. ❌ Backend crashe avec **MÊME ERREUR** : `ModuleNotFoundError: flask_limiter.storage`

#### Pourquoi l'erreur persiste ?

```
Timeline:
21:40 - Push commit c74e247e → Déclenche workflow
21:41 - Job "Build & Push" démarre
21:46 - Job "Deploy" démarre (en parallèle !)
21:46 - Serveur pull l'image Docker Hub
        ❌ PROBLÈME : L'image est encore l'ANCIENNE !
        ⏳ Le build n'a pas encore fini (prend 10-15 min)
21:46 - Backend démarre avec l'ancienne image
        ❌ ModuleNotFoundError: flask_limiter.storage
```

#### Solution

**ATTENDRE** que le workflow termine complètement :

```
Étape 1 : Vérifier que "Build & Push" est terminé (vert) ✅
Étape 2 : Vérifier que l'image est sur Docker Hub
Étape 3 : RE-DÉCLENCHER le déploiement manuellement
```

**Temps estimé** : 10-15 minutes de plus

---

**Date de création** : 2026-01-08
**Créé par** : Analyse automatique des logs de déploiement
**Commit concerné** : `b941db99` → `c74e247e` (workflow en cours)
