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
ssh deploy@138.201.155.201
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
- SERVER_HOST : 138.201.155.201
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

## 🚀 ACTIONS IMMÉDIATES (À FAIRE MAINTENANT)

### Étape 1 : Rebuild l'image (5 min)

```powershell
# Sur Windows PowerShell
cd C:\Users\jasiq\atmr

# Rebuild l'image backend
docker build -t docker.io/djasiqi/atmr-backend:latest ./backend
```

**⏳ Temps de build estimé** : 5-10 minutes

### Étape 2 : Push vers Docker Hub (2 min)

```powershell
# Login à Docker Hub
docker login
# Username: djasiqi
# Password: [votre token Docker Hub]

# Push l'image
docker push docker.io/djasiqi/atmr-backend:latest
```

**⏳ Temps de push estimé** : 2-5 minutes (selon bande passante)

### Étape 3 : Redéployer sur le serveur (3 min)

```bash
# SSH au serveur
ssh deploy@138.201.155.201

cd /home/deploy/atmr

# Pull la nouvelle image
docker-compose -f docker-compose.production.yml pull backend celery-worker celery-beat flower

# Redémarrer les services
docker-compose -f docker-compose.production.yml up -d --force-recreate

# Exécuter les migrations
docker-compose -f docker-compose.production.yml exec backend flask db upgrade

# Vérifier les logs
docker-compose -f docker-compose.production.yml logs -f backend
```

**✅ Attendre le message** : `✅ Backend démarré`

### Étape 4 : Vérifier le service (1 min)

```bash
# Tester l'API
curl -I https://www.lirie.ch/health
# Attendre: HTTP/2 200

# Tester l'API alternative
curl -I https://api.lirie.ch/health
# Attendre: HTTP/2 200
```

---

## ⚠️ AUTRES CORRECTIONS NÉCESSAIRES

Après avoir corrigé le problème Flask-Limiter, il faudra aussi corriger :

### 1. PostgreSQL "role root" (non-bloquant)

**Fichier** : `docker-compose.production.yml`

```diff
postgres:
  healthcheck:
-   test: ["CMD-SHELL", "pg_isready"]
+   test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-atmr_user}"]
    interval: 5s
    timeout: 3s
    retries: 20
    start_period: 90s
```

**Fichier** : `docker-compose.monitoring.yml`

```yaml
# Vérifier que postgres-exporter utilise ${POSTGRES_USER}
postgres-exporter:
  environment:
    - DATA_SOURCE_NAME=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}?sslmode=disable
```

### 2. Nettoyer les conteneurs orphelins

```bash
# Sur le serveur
docker-compose down --remove-orphans
docker-compose -f docker-compose.production.yml up -d
```

---

## 📋 CHECKLIST DE DÉPLOIEMENT

### Avant le déploiement

- [x] Flask-Limiter[redis] présent dans requirements.base.txt
- [ ] Image Docker rebuildée localement
- [ ] Image pushée vers Docker Hub
- [ ] Backup de la base de données effectué (optionnel)

### Pendant le déploiement

- [ ] Pull de la nouvelle image réussi
- [ ] Conteneurs redémarrés avec `--force-recreate`
- [ ] Migrations Alembic exécutées
- [ ] Backend démarre sans erreurs

### Après le déploiement

- [ ] API accessible (`curl https://www.lirie.ch/health`)
- [ ] Logs backend sans erreurs `flask_limiter.storage`
- [ ] Logs PostgreSQL sans `FATAL: role "root"`
- [ ] Socket.IO fonctionne (tester depuis l'app mobile/frontend)

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

**Créé le** : 2026-01-08  
**Auteur** : Analyse automatique des logs de déploiement  
**Statut** : ✅ Prêt pour application
