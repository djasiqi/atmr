# 🔧 PROCÉDURE COMPLÈTE : Correction du déploiement Docker

**Date** : 2026-01-08
**Problème** : `ModuleNotFoundError: No module named 'flask_limiter.storage'`
**Cause racine** : Image Docker Hub obsolète (pas de `Flask-Limiter[redis]`)

---

## 🎯 CAUSE RACINE IDENTIFIÉE

### Le problème

**docker-compose.production.yml** utilise une image prébuiltée depuis Docker Hub :

```yaml
backend:
  image: ${DOCKER_IMAGE:-docker.io/djasiqi/atmr-backend}:${DOCKER_TAG:-latest}
  # ❌ PAS DE SECTION `build:` !
```

**Conséquence** :

1. Le script `deploy-server.sh` lance `docker-compose build --no-cache`
2. Docker Compose ne trouve **AUCUNE section `build:`**
3. Il skip complètement le build → pull juste l'image depuis Docker Hub
4. L'image Docker Hub est **obsolète** (n'a pas `Flask-Limiter[redis]`)

### Pourquoi Flask-Limiter[redis] est nécessaire ?

```python
# backend/ext.py, ligne 167
from flask_limiter.storage import (  # ← Module manquant !
    RedisStorage,
    RedisStorageWithTTL,
)
```

- `Flask-Limiter>=3.0.0` seul ne suffit PAS
- Il faut `Flask-Limiter[redis]>=3.0.0` pour avoir le module `flask_limiter.storage`
- ✅ **DÉJÀ CORRIGÉ** dans `backend/requirements.base.txt` (commit `64ad4671`)
- ❌ **PAS APPLIQUÉ** car image Docker Hub jamais rebuildée

---

## ✅ SOLUTIONS (3 options)

### 🔥 SOLUTION A : Rebuild et push manuel (RECOMMANDÉ - RAPIDE)

**Temps** : 10-15 minutes  
**Difficulté** : ⭐⭐ Moyenne  
**Avantages** :

- ✅ Rapide (correction immédiate)
- ✅ Pas de modification de code
- ✅ Pas besoin de CI/CD

**Étapes** :

#### A.1 - Rebuild l'image localement

```bash
# Sur votre machine locale
cd C:\Users\jasiq\atmr

# Pull les derniers changements (déjà fait via le push récent)
git pull

# Rebuild l'image backend avec Flask-Limiter[redis]
docker build -t docker.io/djasiqi/atmr-backend:latest ./backend

# OU avec un tag de version
docker build -t docker.io/djasiqi/atmr-backend:1.0.0 ./backend
docker tag docker.io/djasiqi/atmr-backend:1.0.0 docker.io/djasiqi/atmr-backend:latest
```

#### A.2 - Push vers Docker Hub

```bash
# Login à Docker Hub (si pas déjà fait)
docker login

# Push l'image
docker push docker.io/djasiqi/atmr-backend:latest

# Si vous avez créé un tag de version
docker push docker.io/djasiqi/atmr-backend:1.0.0
```

#### A.3 - Redéployer sur le serveur

```bash
# Option 1 : Via SSH manuel
ssh deploy@$SERVER_HOST
cd /home/deploy/atmr

# Pull la nouvelle image depuis Docker Hub
docker-compose -f docker-compose.production.yml pull backend celery-worker celery-beat flower

# Redémarrer les services
docker-compose -f docker-compose.production.yml up -d --force-recreate

# Exécuter les migrations
docker-compose -f docker-compose.production.yml exec backend flask db upgrade

# Vérifier les logs
docker-compose -f docker-compose.production.yml logs -f backend

# ✅ Attendre le message : "✅ Backend démarré"
```

```bash
# Option 2 : Via script deploy-server.sh (nécessite modification)
# Voir SOLUTION B ci-dessous
```

---

### 🟡 SOLUTION B : Modifier docker-compose.production.yml pour builder l'image

**Temps** : 15-20 minutes  
**Difficulté** : ⭐⭐⭐ Moyenne-Haute  
**Avantages** :

- ✅ Builds automatiques lors du déploiement
- ✅ Pas besoin de Docker Hub
- ✅ Contrôle total sur l'image

**Inconvénients** :

- ❌ Build sur le serveur (plus lent)
- ❌ Nécessite Git sur le serveur
- ❌ Consomme ressources du serveur

**Modification requise** :

```diff
# docker-compose.production.yml

backend:
- image: ${DOCKER_IMAGE:-docker.io/djasiqi/atmr-backend}:${DOCKER_TAG:-latest}
+ build:
+   context: ./backend
+   dockerfile: Dockerfile
+   args:
+     WITH_POSTGRES: "true"
+ image: atmr-backend:local
  container_name: atmr-backend
  restart: unless-stopped
  # ... reste identique

celery-worker:
- image: ${DOCKER_IMAGE:-docker.io/djasiqi/atmr-backend}:${DOCKER_TAG:-latest}
+ image: atmr-backend:local  # Utiliser la même image que backend
  container_name: atmr-celery-worker
  # ... reste identique

celery-beat:
- image: ${DOCKER_IMAGE:-docker.io/djasiqi/atmr-backend}:${DOCKER_TAG:-latest}
+ image: atmr-backend:local  # Utiliser la même image que backend
  container_name: atmr-celery-beat
  # ... reste identique

flower:
- image: ${DOCKER_IMAGE:-docker.io/djasiqi/atmr-backend}:${DOCKER_TAG:-latest}
+ image: atmr-backend:local  # Utiliser la même image que backend
  container_name: atmr-flower
  # ... reste identique
```

**Après modification** :

```bash
# Commit et push
git add docker-compose.production.yml
git commit -m "fix(deploy): Add build section to docker-compose.production.yml"
git push

# Redéployer via script (qui rebuild maintenant)
bash deploy-server.sh
```

---

### 🟢 SOLUTION C : Créer un workflow GitHub Actions (RECOMMANDÉ - LONG TERME)

**Temps** : 30-45 minutes (setup initial)  
**Difficulté** : ⭐⭐⭐⭐ Élevée  
**Avantages** :

- ✅ Automatique (push → build → deploy)
- ✅ CI/CD complet
- ✅ Image toujours à jour
- ✅ Historique des builds

**Inconvénients** :

- ❌ Nécessite configuration GitHub Secrets
- ❌ Nécessite temps de setup initial

**Étapes** :

#### C.1 - Créer le workflow `.github/workflows/deploy.yml`

```yaml
name: Build & Deploy

on:
  push:
    branches:
      - main
  workflow_dispatch:

env:
  DOCKER_IMAGE: docker.io/djasiqi/atmr-backend
  DOCKER_TAG: latest

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: ./backend
          file: ./backend/Dockerfile
          push: true
          tags: |
            ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
            ${{ env.DOCKER_IMAGE }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /home/deploy/atmr
            git pull origin main
            docker-compose -f docker-compose.production.yml pull
            docker-compose -f docker-compose.production.yml up -d --force-recreate
            docker-compose -f docker-compose.production.yml exec -T backend flask db upgrade
```

#### C.2 - Configurer les GitHub Secrets

```
Aller sur GitHub → Settings → Secrets and variables → Actions → New repository secret

Ajouter :
- DOCKER_USERNAME : votre username Docker Hub
- DOCKER_PASSWORD : votre token Docker Hub
- SERVER_HOST : (défini en local, voir docs/deployment-ssh.md)
- SERVER_USER : deploy
- SSH_PRIVATE_KEY : votre clé SSH privée
```

#### C.3 - Tester le workflow

```bash
# Push un changement pour déclencher le workflow
git commit --allow-empty -m "test: Trigger GitHub Actions workflow"
git push

# Aller sur GitHub → Actions pour voir le workflow en cours
```

---

## 📊 COMPARAISON DES SOLUTIONS

| Critère            | Solution A (Manuel)  | Solution B (Build local) | Solution C (CI/CD)       |
| ------------------ | -------------------- | ------------------------ | ------------------------ |
| **Rapidité**       | ⭐⭐⭐⭐⭐ 10-15 min | ⭐⭐⭐⭐ 15-20 min       | ⭐⭐⭐ 30-45 min (setup) |
| **Complexité**     | ⭐⭐ Moyenne         | ⭐⭐⭐ Moyenne-Haute     | ⭐⭐⭐⭐ Élevée          |
| **Automatisation** | ❌ Manuel            | ⭐⭐⭐ Semi-auto         | ⭐⭐⭐⭐⭐ Full auto     |
| **Long terme**     | ❌ Répétitif         | ⭐⭐⭐⭐ Bon             | ⭐⭐⭐⭐⭐ Excellent     |
| **Recommandé**     | Pour fix urgent      | Pour petites équipes     | Pour production          |

---

## 🎯 RECOMMANDATION

### Pour correction IMMÉDIATE (aujourd'hui)

**➡️ Utilisez SOLUTION A (Rebuild manuel)**

### Pour amélioration LONG TERME (cette semaine)

**➡️ Implémentez SOLUTION C (GitHub Actions CI/CD)**

---

## 🚀 ACTIONS IMMÉDIATES (CI/CD Automatisé)

### ⏳ MAINTENANT : Attendre le workflow GitHub Actions (10-15 min)

**❌ NE RIEN FAIRE MANUELLEMENT !** GitHub Actions fait tout automatiquement.

```
Étape 1 : Vérifier le workflow
Aller sur : https://github.com/djasiqi/atmr/actions

Étape 2 : Trouver le Run #251
Workflow : "Build & Deploy"
Commit : c74e247e (feat: Activate GitHub Actions workflow)

Étape 3 : Vérifier l'état
Job "Build & Push" : ⏳ EN COURS (5-10 min)
  - Build Dockerfile.production
  - Scan Trivy (sécurité)
  - Push vers Docker Hub

Job "Deploy" : ❌ ÉCHOUÉ (premier essai avec ancienne image)
  - Normal ! L'image n'était pas encore prête

Temps restant estimé : 10-15 min
```

---

### 🔄 APRÈS LE BUILD : Re-déclencher le déploiement (3 options)

**Attendez que le Job "Build & Push" soit ✅ VERT avant de continuer !**

#### **Option A : Via GitHub Actions UI** (⭐ RECOMMANDÉ - Le plus simple)

```
1. Aller sur : https://github.com/djasiqi/atmr/actions
2. Cliquer sur "Build & Deploy" dans la liste des workflows
3. Cliquer sur "Run workflow" (bouton vert à droite)
4. Laisser tous les champs par défaut
5. Cliquer sur "Run workflow" (confirmer)
6. Attendre 5-10 min (nouveau déploiement avec la nouvelle image)
7. Vérifier que tout est ✅ VERT
```

**⏳ Temps total** : 5-10 minutes

---

#### **Option B : Via Git Push** (Déclenche automatiquement)

```bash
# Sur votre PC Windows
cd C:\Users\jasiq\atmr

# Commit vide pour déclencher le workflow
git commit --allow-empty -m "chore: Trigger re-deployment with new Docker image"
git push origin main

# Le workflow se lance automatiquement
# Aller sur GitHub → Actions pour suivre
```

**⏳ Temps total** : 15-20 minutes (build + deploy)

---

#### **Option C : Via SSH Manuel** (Si urgent)

```bash
# SSH au serveur
ssh deploy@$SERVER_HOST

cd /srv/atmr

# Vérifier que la NOUVELLE image existe sur Docker Hub
docker manifest inspect djasiqi/atmr-backend:latest
# Vérifier la date : doit être 2026-01-08 (aujourd'hui)

# Pull la NOUVELLE image
docker-compose -f docker-compose.production.yml pull backend celery-worker celery-beat flower

# Redémarrer avec la nouvelle image
docker-compose -f docker-compose.production.yml up -d --force-recreate

# Exécuter les migrations
docker-compose -f docker-compose.production.yml exec backend flask db upgrade

# Vérifier les logs
docker-compose -f docker-compose.production.yml logs -f backend

# ✅ Attendre le message : "✅ Backend démarré"
# ✅ SANS erreur "ModuleNotFoundError: flask_limiter.storage"
```

**⏳ Temps total** : 5 minutes

---

### ✅ VÉRIFICATION FINALE : Tester le service (2 min)

```bash
# Tester l'API (depuis votre PC ou le serveur)
curl -I https://www.lirie.ch/health
# ✅ Attendu : HTTP/2 200

curl -I https://api.lirie.ch/health
# ✅ Attendu : HTTP/2 200

# Vérifier les logs backend
ssh deploy@$SERVER_HOST
cd /srv/atmr
docker-compose -f docker-compose.production.yml logs --tail=50 backend

# ✅ Vérifier qu'il n'y a PLUS d'erreur flask_limiter.storage
```

---

## ✅ CORRECTIONS DÉJÀ APPLIQUÉES

Ces corrections ont été faites lors de l'activation du CI/CD :

### 1. ✅ PostgreSQL "role root" - CORRIGÉ

**Fichier** : `docker-compose.production.yml` (commit `a987f774`)

```yaml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-atmr_user}"]
    # ✅ Corrigé : utilise maintenant ${POSTGRES_USER}
```

**Fichier** : `docker-compose.monitoring.yml` (déjà correct)

```yaml
postgres-exporter:
  environment:
    - DATA_SOURCE_NAME=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}?sslmode=disable
    # ✅ Correct : utilise ${POSTGRES_USER}
```

### 2. ⏳ Conteneurs orphelins - Se nettoiera automatiquement

Les warnings "orphan containers" sont normaux pendant le déploiement. Ils se résoudront automatiquement au prochain redémarrage.

---

## 📋 CHECKLIST DE DÉPLOIEMENT (Automatisé via GitHub Actions)

### ✅ Avant le déploiement (déjà fait !)

- [x] Flask-Limiter[redis] présent dans requirements.base.txt
- [x] Workflow GitHub Actions activé
- [x] Push vers main effectué (commit c74e247e)
- [x] Secrets GitHub configurés (47/47)

### ⏳ Pendant le build (GitHub Actions - AUTOMATIQUE)

- [x] Workflow déclenché (Run #251)
- [ ] ⏳ Job "Build & Push" en cours (10-15 min)
  - [ ] Build Dockerfile.production
  - [ ] Scan Trivy (sécurité)
  - [ ] Push vers Docker Hub
- [ ] ⏳ Nouvelle image disponible sur Docker Hub

### 🔄 Re-déploiement (À FAIRE après le build)

**Option 1 : Automatique** (laisser le workflow terminer)

- [ ] Attendre que le workflow termine complètement
- [ ] Vérifier que tout est vert ✅

**Option 2 : Manuel** (si Option 1 échoue)

- [ ] Vérifier image sur Docker Hub mise à jour
- [ ] Re-déclencher le déploiement (GitHub Actions → Run workflow)
- [ ] OU SSH au serveur + pull + restart

### ✅ Après le déploiement (vérifications finales)

- [ ] API accessible (`curl https://www.lirie.ch/health` → HTTP/2 200)
- [ ] Logs backend SANS erreur `flask_limiter.storage` ✅
- [ ] Logs PostgreSQL SANS `FATAL: role "root"` ✅
- [ ] Socket.IO fonctionne (tester app mobile/frontend)

---

## 🛟 ROLLBACK EN CAS DE PROBLÈME

Si le déploiement échoue :

```bash
# Sur le serveur
cd /home/deploy/atmr

# Revenir à l'image précédente (si taggée)
docker-compose -f docker-compose.production.yml down
docker pull docker.io/djasiqi/atmr-backend:previous
docker tag docker.io/djasiqi/atmr-backend:previous docker.io/djasiqi/atmr-backend:latest
docker-compose -f docker-compose.production.yml up -d
```

**OU**

```bash
# Rollback Git
git log --oneline -5  # Trouver le commit précédent
git revert <commit_sha>
git push
# Puis redéployer
```

---

## 📝 NOTES IMPORTANTES

### Pourquoi l'image n'a pas été rebuildée automatiquement ?

1. **Pas de section `build:`** dans `docker-compose.production.yml`
2. **Pas de workflow GitHub Actions** pour rebuilder automatiquement
3. **Script deploy-server.sh** lance `docker-compose build` mais ça ne fait rien car pas de `build:`

### Pourquoi Flask-Limiter[redis] ?

- `Flask-Limiter>=3.0.0` seul installe **juste le core**
- `Flask-Limiter[redis]>=3.0.0` installe **core + redis-py + extras**
- Le module `flask_limiter.storage` fait partie des **extras Redis**

### Prochaines étapes (après correction)

1. ✅ Implémenter Solution C (CI/CD GitHub Actions)
2. ✅ Ajouter tests pour vérifier les imports critiques
3. ✅ Configurer alerting si backend down
4. ✅ Documenter la procédure de déploiement

---

---

## ⚠️ MISE À JOUR IMPORTANTE : 2026-01-08 21:46 UTC

### 🔄 Workflow GitHub Actions EN COURS (Run #251)

**Problème identifié** :

Le job "Deploy" s'est lancé **AVANT** que le job "Build & Push" termine !

**Résultat** :

- ❌ Serveur a pull l'ANCIENNE image Docker Hub (obsolète)
- ❌ Backend crashe avec **MÊME ERREUR** : `ModuleNotFoundError: flask_limiter.storage`
- ⏳ Job "Build & Push" encore en cours (construction de la NOUVELLE image avec Flask-Limiter[redis])

**Solution IMMÉDIATE** :

```bash
# ÉTAPE 1 : Attendre que le workflow termine (10-15 min)
# Aller sur : https://github.com/djasiqi/atmr/actions/runs/251
# Vérifier que "Build & Push" est ✅ VERT (terminé avec succès)

# ÉTAPE 2 : Vérifier que l'image est mise à jour sur Docker Hub
docker manifest inspect djasiqi/atmr-backend:latest
# Vérifier la date de création (doit être aujourd'hui)

# ÉTAPE 3 : RE-DÉCLENCHER le déploiement
# Option A : Via GitHub Actions UI
#   1. GitHub → Actions → Build & Deploy
#   2. Run workflow → skip_deploy: false → Run workflow

# Option B : Via SSH manuel sur le serveur
ssh deploy@$SERVER_HOST
cd /srv/atmr
docker-compose -f docker-compose.production.yml pull backend celery-worker celery-beat flower
docker-compose -f docker-compose.production.yml up -d --force-recreate
docker-compose -f docker-compose.production.yml exec backend flask db upgrade
```

**Timeline estimée** :

```
21:46 - ❌ Premier déploiement échoué (image obsolète)
21:55 - ✅ Build & Push terminé (nouvelle image disponible)
22:00 - ✅ Re-déploiement réussi (backend démarre correctement)
```

---

**Créé le** : 2026-01-08  
**Mis à jour** : 2026-01-08 21:46 UTC  
**Auteur** : Analyse automatique des logs de déploiement  
**Statut** : ⏳ En attente de la fin du build (10-15 min)
