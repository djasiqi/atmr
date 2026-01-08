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

### 2. 🔴 **PostgreSQL : role "root" does not exist**

**Erreur** :

```
FATAL:  role "root" does not exist
```

**Fréquence** : Se répète toutes les 5 secondes

**Occurrences** :

```
2026-01-08 21:25:14.586 UTC [39] FATAL:  role "root" does not exist
2026-01-08 21:25:19.667 UTC [46] FATAL:  role "root" does not exist
2026-01-08 21:25:24.767 UTC [61] FATAL:  role "root" does not exist
2026-01-08 21:25:29.871 UTC [68] FATAL:  role "root" does not exist
2026-01-08 21:25:34.970 UTC [76] FATAL:  role "root" does not exist
2026-01-08 21:25:40.065 UTC [83] FATAL:  role "root" does not exist
2026-01-08 21:25:45.159 UTC [91] FATAL:  role "root" does not exist
```

**Impact** :

- ⚠️ Tentatives de connexion échouées
- ⚠️ Logs pollués
- ⚠️ Possible problème de health check

**Cause racine** :

- Un service tente de se connecter à PostgreSQL avec l'utilisateur `root`
- Probablement un health check ou monitoring mal configuré

**Localisation probable** :

- `docker-compose.monitoring.yml` - postgres-exporter
- `docker-compose.production.yml` - health checks PostgreSQL

**Solution** :

```yaml
# Dans docker-compose.production.yml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-atmr_user}"]
    # ❌ NE PAS UTILISER: pg_isready -U root

# Dans docker-compose.monitoring.yml
postgres-exporter:
  environment:
    - DATA_SOURCE_NAME=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}?sslmode=disable
    # ❌ NE PAS UTILISER: root comme utilisateur
```

**Actions requises** :

1. ❌ Vérifier `docker-compose.production.yml` - section `postgres.healthcheck`
2. ❌ Vérifier `docker-compose.monitoring.yml` - `postgres-exporter.environment.DATA_SOURCE_NAME`
3. ❌ S'assurer que toutes les connexions utilisent `${POSTGRES_USER}` (défini dans `.env.production`)

**Priorité** : 🟡 MOYENNE (Non bloquant mais à corriger)

---

### 3. 🔴 **Migrations Alembic échouent**

**Erreur** :

```
Error: No such command 'db'.
```

**Cause** :

- Conséquence directe de l'erreur #1 (flask_limiter.storage)
- Flask CLI ne peut pas charger l'app à cause de l'import error

**Commandes qui échouent** :

```bash
flask db current
flask db upgrade
```

**Impact** :

- ❌ Base de données potentiellement pas à jour
- ❌ Nouvelles migrations non appliquées

**Solution** :

- Corriger l'erreur #1 d'abord
- Puis les commandes `flask db` fonctionneront

**Priorité** : 🔥 URGENTE (Dépend de #1)

---

## ⚠️ AVERTISSEMENTS (Non-bloquants mais à surveiller)

### 4. 🟡 **Conteneurs orphelins détectés**

**Message** :

```
level=warning msg="Found orphan containers ([***-backend ***-celery-worker ***-flower ***-celery-beat ***-postgres ***-redis]) for this project."
```

**Cause** :

- Anciens conteneurs non nettoyés après modifications de docker-compose

**Solution** :

```bash
docker-compose down --remove-orphans
```

**Priorité** : 🟢 BASSE (Cosmétique)

---

### 5. 🟡 **Secrets Vault non trouvés (warnings)**

**Messages** :

```
[4.1 Vault] Aucune authentification configurée, désactivation
[4.1 Vault] Secret non trouvé: JWT_LEGACY_SECRET_KEYS (path=dev/jwt/legacy_secret_keys, key=keys)
[4.1 Vault] Secret non trouvé: JWT_LEGACY_SECRET_KEYS (path=prod/jwt/legacy_secret_keys, key=keys)
```

**Impact** :

- ℹ️ Pas de Vault configuré en production
- ℹ️ L'app utilise les secrets depuis les variables d'environnement (fallback)
- ℹ️ Fonctionnalité dégradée mais non bloquante

**Cause** :

- Vault non configuré sur ce serveur
- C'est attendu si vous n'utilisez pas HashiCorp Vault

**Solution** :

- Si vous voulez utiliser Vault : configurer `VAULT_ADDR`, `VAULT_TOKEN`, etc.
- Sinon : ignorer ces warnings (comportement normal)

**Priorité** : 🟢 BASSE (Optionnel)

---

### 6. 🟡 **Modèle ML de prédiction de retard non trouvé**

**Message** :

```
⚠️  Modèle de prédiction de retard non trouvé
```

**Impact** :

- ⚠️ Fonctionnalité ML désactivée
- ℹ️ App fonctionne sans prédictions ML

**Cause** :

- Le fichier `delay_predictor.pkl` n'est pas au bon emplacement ou corrompu
- Permissions ML corrigées mais modèle manquant

**Solution** :

```bash
# Sur le serveur
docker exec ***-backend ls -la /app/ml_models/
# Vérifier si delay_predictor.pkl existe

# Si manquant, entraîner un nouveau modèle ou copier depuis backup
```

**Priorité** : 🟢 BASSE (Fonctionnalité optionnelle)

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
