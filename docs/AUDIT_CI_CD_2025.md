# 🔒 Audit CI/CD – Rapport Professionnel (ATMR)

**Date d'analyse :** 2025-11-22  
**Pipeline analysé :** GitHub Actions - Build & Deploy  
**Workflow :** `.github/workflows/deploy.yml`  
**Exécution :** Run #19595024881  
**Durée totale :** ~6 minutes

---

## 1. Résumé Exécutif

Le pipeline CI/CD ATMR présente **un problème critique bloquant** lors du déploiement : les migrations Flask échouent systématiquement en raison de variables d'environnement manquantes (`SQLALCHEMY_DATABASE_URI`). Le build Docker et le scan Trivy fonctionnent correctement, mais le déploiement SSH échoue à chaque exécution. Des optimisations significatives sont possibles au niveau du cache Docker, de la gestion des secrets, et de la configuration Trivy. Le pipeline nécessite des correctifs immédiats pour la configuration des variables d'environnement dans le script de déploiement SSH, ainsi qu'une amélioration de la gestion du cache GitHub Actions pour éviter les conflits.

**Risques majeurs identifiés :**
- 🔴 **CRITIQUE** : Échec systématique des migrations Flask (déploiement bloqué)
- 🟠 **HAUTE** : Conflits de cache GitHub Actions (QEMU)
- 🟠 **HAUTE** : Variables d'environnement non propagées correctement au conteneur backend
- 🟡 **MOYENNE** : Build Docker non optimisé (pas de cache, temps élevé)
- 🟡 **MOYENNE** : Scan Trivy avec warnings (SBOM tiers, sévérités)

**Recommandation globale prioritaire :** Corriger immédiatement la configuration des variables d'environnement dans le script de déploiement SSH pour permettre l'exécution des migrations Flask.

---

## 2. Vue d'Ensemble du Pipeline

### 2.1 Type de Rapport Détecté

**Type :** Pipeline CI/CD complet (Build → Scan → Deploy)  
**Frameworks/Technos identifiés :**
- GitHub Actions (Runner Ubuntu 24.04.3 LTS)
- Docker Buildx v0.29.1 (BuildKit v0.26.2)
- Docker Engine 28.0.4
- Python 3.11 Slim Bookworm
- Trivy v0.67.2
- Docker Compose v2.38.2
- Flask + SQLAlchemy (migrations)
- PostgreSQL 15.14

### 2.2 Étapes Détectées

1. ✅ **Set up job** - Configuration runner
2. ✅ **Build appleboy/scp-action** - Container action SCP
3. ✅ **Build appleboy/ssh-action** - Container action SSH
4. ✅ **Checkout** - Récupération code source
5. ✅ **Set up QEMU** - Emulation multi-arch
6. ✅ **Set up Docker Buildx** - Builder multi-platform
7. ✅ **Login Docker Hub** - Authentification registry
8. ✅ **Build & Push backend image** - Construction et push image Docker
9. ✅ **Scan image with Trivy** - Analyse sécurité
10. ✅ **Copy compose files to server** - Transfert SCP
11. ❌ **Deploy via SSH** - **ÉCHEC** (migrations Flask)

### 2.3 Analyse Générale du Workflow

Le pipeline suit une architecture classique CI/CD avec build, scan sécurité, et déploiement. La partie build fonctionne correctement (~3 minutes), le scan Trivy s'exécute sans erreur (~1 minute), mais le déploiement échoue systématiquement lors de l'exécution des migrations Flask.

**Points positifs :**
- Utilisation de Docker Buildx pour builds optimisés
- Scan sécurité Trivy intégré
- Multi-arch support (QEMU)
- Healthchecks PostgreSQL implémentés

**Points négatifs :**
- Variables d'environnement non propagées au conteneur backend
- Pas de cache Docker optimisé
- Conflits de cache GitHub Actions
- Build Docker long (pas de cache layers)

---

## 3. Problèmes Détectés (Classés par Criticité)

### 🔥 Critique

#### 3.1 Échec Systématique des Migrations Flask

**Description :**  
Les migrations Flask échouent avec l'erreur `RuntimeError: Either 'SQLALCHEMY_DATABASE_URI' or 'SQLALCHEMY_BINDS' must be set.` lors de l'exécution de `flask db upgrade` dans le conteneur backend.

**Extrait du rapport :**
```
2025-11-22T11:53:06.0501583Z err: RuntimeError: Either 'SQLALCHEMY_DATABASE_URI' or 'SQLALCHEMY_BINDS' must be set.
2025-11-22T11:53:06.4722122Z err: + echo '❌ Erreur lors de l'\''exécution des migrations'
```

**Impact :**  
- Déploiement complètement bloqué
- Application non fonctionnelle en production
- Base de données non migrée
- Service backend non démarré correctement

**Cause racine :**  
Les variables d'environnement sont exportées dans le script SSH, mais ne sont **pas propagées au conteneur Docker backend** lors de l'exécution de `docker compose exec`. Le script SSH exporte les variables dans le shell, mais Docker Compose ne les transmet pas automatiquement au conteneur.

**Correctif actionnable :**

**Patch GitHub Actions - Script SSH :**

```yaml
# Dans l'étape "Deploy via SSH"
script: |
  cd /srv/***
  set -o errexit -o nounset -o pipefail -x
  
  # Export des variables pour le shell
  export APP_ENCRYPTION_KEY_B64="${{ env.APP_ENCRYPTION_KEY_B64 }}"
  export SECRET_KEY="${{ env.SECRET_KEY }}"
  export JWT_SECRET_KEY="${{ env.JWT_SECRET_KEY }}"
  export POSTGRES_PASSWORD="${{ env.POSTGRES_PASSWORD }}"
  export POSTGRES_USER="${{ env.POSTGRES_USER }}"
  export POSTGRES_DB="${{ env.POSTGRES_DB }}"
  export MAIL_PASSWORD="${{ env.MAIL_PASSWORD }}"
  export SENTRY_DSN="${{ env.SENTRY_DSN }}"
  export DOCKER_IMAGE="${{ env.DOCKER_IMAGE }}"
  export DOCKER_TAG="${{ env.DOCKER_TAG }}"
  
  # Construction de SQLALCHEMY_DATABASE_URI
  export SQLALCHEMY_DATABASE_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
  
  echo "🔄 Pull de l'image Docker..."
  docker compose -f docker-compose.production.yml pull
  
  echo "🔄 Arrêt des conteneurs existants..."
  docker compose -f docker-compose.production.yml down --remove-orphans || true
  
  echo "🔄 Démarrage des services..."
  docker compose -f docker-compose.production.yml up -d
  
  echo "⏳ Attente que PostgreSQL soit prêt..."
  echo "📊 Vérification de l'état du conteneur PostgreSQL..."
  for i in $(seq 1 60); do
    POSTGRES_STATUS=$(docker compose -f docker-compose.production.yml ps postgres --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$POSTGRES_STATUS" = "running" ]; then
      HEALTH=$(docker inspect --format='{{.State.Health.Status}}' ***-postgres 2>/dev/null || echo "none")
      if [ "$HEALTH" = "healthy" ]; then
        if docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > /dev/null 2>&1; then
          echo "✅ PostgreSQL est prêt et healthy"
          break
        fi
      elif [ "$HEALTH" = "unhealthy" ]; then
        echo "⚠️  PostgreSQL est unhealthy, affichage des logs..."
        docker compose -f docker-compose.production.yml logs postgres | tail -50
      fi
    elif [ "$POSTGRES_STATUS" = "exited" ] || [ "$POSTGRES_STATUS" = "dead" ]; then
      echo "❌ Le conteneur PostgreSQL a échoué, affichage des logs..."
      docker compose -f docker-compose.production.yml logs postgres | tail -50
      exit 1
    fi
    if [ $i -eq 60 ]; then
      echo "❌ Timeout: PostgreSQL n'est pas prêt après 120 secondes"
      docker compose -f docker-compose.production.yml ps postgres
      docker compose -f docker-compose.production.yml logs postgres | tail -100
      exit 1
    fi
    echo "  Tentative $i/60 (État: ${POSTGRES_STATUS:-unknown}, Health: ${HEALTH:-none})..."
    sleep 2
  done
  
  echo "🔄 Exécution des migrations de base de données..."
  # CORRECTION : Passer les variables d'environnement explicitement
  docker compose -f docker-compose.production.yml exec -T \
    -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI}" \
    -e POSTGRES_USER="${POSTGRES_USER}" \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
    -e POSTGRES_DB="${POSTGRES_DB}" \
    -e APP_ENCRYPTION_KEY_B64="${APP_ENCRYPTION_KEY_B64}" \
    -e SECRET_KEY="${SECRET_KEY}" \
    -e JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
    -e MAIL_PASSWORD="${MAIL_PASSWORD}" \
    -e SENTRY_DSN="${SENTRY_DSN}" \
    backend flask db upgrade || {
    echo "❌ Erreur lors de l'exécution des migrations"
    echo "---- Diagnostics (migration failed) ----"
    docker compose -f docker-compose.production.yml exec -T backend flask db current || true
    docker compose -f docker-compose.production.yml exec -T backend flask db heads || true
    docker compose -f docker-compose.production.yml logs backend | tail -50 || true
    exit 1
  }
  echo "✅ Migrations appliquées avec succès"
  echo "✅ Déploiement terminé"
```

**Alternative : Utiliser docker-compose.production.yml avec env_file**

**Patch docker-compose.production.yml :**

```yaml
services:
  backend:
    # ... autres configurations ...
    environment:
      - SQLALCHEMY_DATABASE_URI=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
      - APP_ENCRYPTION_KEY_B64=${APP_ENCRYPTION_KEY_B64}
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - MAIL_PASSWORD=${MAIL_PASSWORD}
      - SENTRY_DSN=${SENTRY_DSN}
```

**Note :** La meilleure solution est de définir `SQLALCHEMY_DATABASE_URI` directement dans le `docker-compose.production.yml` pour éviter toute dépendance aux variables shell.

---

### ⚠️ Haute

#### 3.2 Conflit de Cache GitHub Actions (QEMU)

**Description :**  
Le cache de l'image QEMU échoue à sauvegarder avec l'erreur "Unable to reserve cache with key docker.io--tonistiigi--binfmt-***-linux-x64, another job may be creating this cache."

**Extrait du rapport :**
```
2025-11-22T11:53:19.4624649Z Failed to save: Unable to reserve cache with key docker.io--tonistiigi--binfmt-***-linux-x64, another job may be creating this cache.
```

**Impact :**  
- Cache QEMU non sauvegardé (perte de performance)
- Risque de ralentissement des builds suivants
- Pas d'impact fonctionnel direct

**Cause racine :**  
Conflit de cache GitHub Actions lorsque plusieurs jobs s'exécutent simultanément et tentent de créer le même cache.

**Correctif actionnable :**

**Patch GitHub Actions :**

```yaml
- name: Set up QEMU
  uses: docker/setup-qemu-action@v3
  with:
    image: docker.io/tonistiigi/binfmt:latest
    platforms: all
    cache-image: true
  # Ajouter un timeout et retry logic
  continue-on-error: true  # Ne pas faire échouer le job si le cache échoue

# Ou utiliser une clé de cache unique par job
- name: Set up QEMU
  uses: docker/setup-qemu-action@v3
  with:
    image: docker.io/tonistiigi/binfmt:latest
    platforms: all
    cache-image: true
    cache-key: qemu-binfmt-${{ github.run_id }}-${{ github.run_attempt }}
```

---

#### 3.3 Warnings Trivy sur SBOM Tiers

**Description :**  
Trivy émet des warnings concernant l'utilisation de SBOM tiers qui peuvent conduire à une détection de vulnérabilités imprécise.

**Extrait du rapport :**
```
2025-11-22T11:51:28.5012299Z WARN	Third-party SBOM may lead to inaccurate vulnerability detection
2025-11-22T11:51:28.5014064Z WARN	Recommend using Trivy to generate SBOMs
```

**Impact :**  
- Risque de faux positifs/négatifs dans les scans
- Détection de vulnérabilités potentiellement incomplète
- Impact sécurité moyen

**Cause racine :**  
L'image Docker utilise un SBOM généré par un outil tiers au lieu d'utiliser Trivy pour générer le SBOM.

**Correctif actionnable :**

**Patch GitHub Actions - Génération SBOM avec Trivy :**

```yaml
- name: Generate SBOM with Trivy
  uses: aquasecurity/trivy-action@master
  with:
    version: v0.67.2
    scan-type: 'fs'
    scan-ref: './backend'
    format: 'cyclonedx'
    output: 'sbom-cyclonedx.json'
    cache-dir: /home/runner/work/***/***/.cache/trivy

- name: Build & Push backend image
  uses: docker/build-push-action@v5
  with:
    context: ./backend
    file: ./backend/Dockerfile.production
    build-args: |
      WITH_RL=false
      TRIVY_SBOM=sbom-cyclonedx.json  # Passer le SBOM au build
    push: true
    tags: ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

**Patch Dockerfile.production :**

```dockerfile
# Ajouter le SBOM dans l'image
COPY --from=sbom-generator /sbom-cyclonedx.json /app/sbom-cyclonedx.json
```

---

### ⚙️ Moyenne

#### 3.4 Build Docker Non Optimisé (Pas de Cache)

**Description :**  
Le build Docker ne utilise pas de cache GitHub Actions, ce qui rallonge significativement le temps de build (~3 minutes).

**Extrait du rapport :**
```yaml
# Pas de cache-from/cache-to dans le build
```

**Impact :**  
- Temps de build élevé (~3 minutes)
- Consommation de ressources GitHub Actions inutile
- Coût potentiellement plus élevé

**Cause racine :**  
Absence de configuration `cache-from` et `cache-to` dans l'action `docker/build-push-action`.

**Correctif actionnable :**

**Patch GitHub Actions :**

```yaml
- name: Build & Push backend image
  uses: docker/build-push-action@v5
  with:
    context: ./backend
    file: ./backend/Dockerfile.production
    build-args: WITH_RL=false
    push: true
    tags: ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
    # OPTIMISATION : Ajouter le cache GitHub Actions
    cache-from: type=gha
    cache-to: type=gha,mode=max
    # OPTIMISATION : Cache inline pour layers Docker
    cache-from: |
      type=gha
      type=registry,ref=${{ env.DOCKER_IMAGE }}:buildcache
    cache-to: |
      type=gha,mode=max
      type=registry,ref=${{ env.DOCKER_IMAGE }}:buildcache,mode=max
```

**Gain estimé :** Réduction du temps de build de ~3 minutes à ~1 minute (si cache hit).

---

#### 3.5 Warnings Pip sur Exécution Root

**Description :**  
Pip émet des warnings lors de l'installation des packages car l'exécution se fait en tant que root.

**Extrait du rapport :**
```
2025-11-22T11:48:38.2393057Z WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager.
```

**Impact :**  
- Risque de permissions incorrectes
- Conflits potentiels avec le gestionnaire de paquets système
- Impact sécurité faible (conteneur isolé)

**Cause racine :**  
Le Dockerfile exécute pip en tant que root au lieu d'utiliser un utilisateur non-privilégié.

**Correctif actionnable :**

**Patch Dockerfile.production :**

```dockerfile
# Créer un utilisateur non-privilégié
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Installer les dépendances en tant que root (nécessaire)
RUN python -m pip install --upgrade pip && \
    pip install --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels /root/.cache/pip

# Changer vers l'utilisateur non-privilégié
USER appuser

# Définir le working directory
WORKDIR /app
```

**Note :** Pour les builds, l'exécution en root est acceptable, mais pour la production, utiliser un utilisateur non-privilégié est une bonne pratique.

---

### 🟩 Basse

#### 3.6 Git Hint sur Master Branch

**Description :**  
Git émet un hint concernant l'utilisation de 'master' comme nom de branche par défaut.

**Extrait du rapport :**
```
2025-11-22T11:47:28.2387227Z hint: Using 'master' as the name for the initial branch. This default branch name is subject to change.
```

**Impact :**  
- Aucun impact fonctionnel
- Message informatif uniquement

**Correctif actionnable :**

**Patch GitHub Actions :**

```yaml
- name: Checkout code
  uses: actions/checkout@v4
  with:
    # Supprimer le hint en configurant la branche par défaut
    fetch-depth: 1
    # Le hint disparaîtra automatiquement avec checkout@v4
```

**Note :** Ce warning est cosmétique et n'affecte pas le fonctionnement.

---

#### 3.7 Warnings Trivy sur Sévérités Multi-Vendeurs

**Description :**  
Trivy émet un warning concernant l'utilisation de sévérités provenant d'autres vendeurs.

**Extrait du rapport :**
```
2025-11-22T11:51:28.5830678Z WARN	Using severities from other vendors for some vulnerabilities.
```

**Impact :**  
- Aucun impact fonctionnel
- Information sur la source des sévérités

**Correctif actionnable :**

**Patch GitHub Actions - Configuration Trivy :**

```yaml
- name: Scan image with Trivy
  uses: aquasecurity/trivy-action@master
  with:
    version: v0.67.2
    image-ref: ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
    format: sarif
    output: trivy-results.sarif
    ignore-unfixed: true
    vuln-type: os,library
    scan-type: image
    severity: UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL
    # Supprimer le warning en utilisant uniquement les sévérités Trivy
    trivyignores: .trivyignore  # Créer un fichier .trivyignore si nécessaire
```

**Note :** Ce warning est informatif et n'indique pas un problème de sécurité.

---

## 4. Analyse par Étape du Pipeline

### 4.1 Checkout

**Statut :** ✅ Réussi  
**Durée :** ~2 secondes  
**Anomalies :** Aucune  
**Optimisations possibles :**
- Utiliser `fetch-depth: 0` uniquement si nécessaire (actuellement `fetch-depth: 1` est optimal)
- Considérer `sparse-checkout` si le repository est très volumineux

---

### 4.2 Setup QEMU

**Statut :** ⚠️ Partiellement réussi (cache échoue)  
**Durée :** ~6 secondes  
**Anomalies :**
- Cache QEMU non sauvegardé (conflit)
- Image QEMU chargée depuis cache (bon)

**Optimisations possibles :**
- Implémenter retry logic pour le cache
- Utiliser une clé de cache unique par job

---

### 4.3 Setup Buildx

**Statut :** ✅ Réussi  
**Durée :** ~3 secondes  
**Anomalies :** Aucune  
**Optimisations possibles :**
- Considérer `keep-state: true` pour réutiliser le builder entre jobs (si applicable)
- Utiliser `driver-opts: network=host` pour améliorer les performances réseau

---

### 4.4 Login Docker Hub

**Statut :** ✅ Réussi  
**Durée :** <1 seconde  
**Anomalies :** Aucune  
**Optimisations possibles :** Aucune

---

### 4.5 Build & Push

**Statut :** ✅ Réussi  
**Durée :** ~3 minutes  
**Taille image :** Non spécifiée dans les logs (à vérifier)  
**Multi-arch :** Non configuré (build uniquement pour linux/amd64)  
**Cache :** ❌ Aucun cache utilisé  
**Erreurs :** Aucune  
**Warnings :**
- Pip exécuté en root (3 occurrences)
- Location '/wheels' ignorée (3 occurrences - normal, c'est un path local)

**Optimisations Dockerfile :**

```dockerfile
# OPTIMISATION 1 : Utiliser BuildKit cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --no-index --find-links=/wheels -r requirements.txt

# OPTIMISATION 2 : Multi-stage build optimisé
FROM python:3.11-slim-bookworm AS builder
# ... build wheels ...

FROM python:3.11-slim-bookworm AS runtime
COPY --from=builder /wheels /wheels
# ... install from wheels ...

# OPTIMISATION 3 : Utiliser un utilisateur non-privilégié
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

**Optimisations GitHub Actions :**

```yaml
- name: Build & Push backend image
  uses: docker/build-push-action@v5
  with:
    context: ./backend
    file: ./backend/Dockerfile.production
    build-args: WITH_RL=false
    push: true
    tags: ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
    # OPTIMISATION : Cache GitHub Actions
    cache-from: type=gha
    cache-to: type=gha,mode=max
    # OPTIMISATION : Multi-arch build (si nécessaire)
    platforms: linux/amd64,linux/arm64
```

**Gain estimé :** Réduction du temps de build de ~3 minutes à ~1 minute (avec cache).

---

### 4.6 Scan Trivy

**Statut :** ✅ Réussi  
**Durée :** ~1 minute  
**Vulnérabilités détectées :** Non spécifiées dans les logs (à vérifier dans le SARIF)  
**Packages OS :** Debian 12.12 (164 packages)  
**Libraries Python :** 1 fichier détecté  
**Risques :** Aucun risque critique identifié dans les logs  
**Warnings :**
- SBOM tiers (imprécision possible)
- Sévérités multi-vendeurs

**Correctifs :**

**Patch Trivy - Configuration améliorée :**

```yaml
- name: Scan image with Trivy
  uses: aquasecurity/trivy-action@master
  with:
    version: v0.67.2
    image-ref: ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
    format: sarif
    output: trivy-results.sarif
    ignore-unfixed: true
    vuln-type: os,library
    scan-type: image
    severity: CRITICAL,HIGH  # Seulement CRITICAL et HIGH pour éviter le bruit
    cache-dir: /home/runner/work/***/***/.cache/trivy
    list-all-pkgs: false
    cache: true
    # OPTIMISATION : Générer le SBOM avec Trivy
    generate-sbom: true
    sbom-format: cyclonedx
```

**Patch .trivyignore (si nécessaire) :**

```text
# Ignorer les vulnérabilités connues et acceptées
CVE-2024-XXXXX  # Raison : vulnérabilité acceptée, patch non disponible
```

---

### 4.7 SCP / Transfert vers Serveur

**Statut :** ✅ Réussi  
**Durée :** ~9 secondes  
**Risques SSH :** Aucun (utilisation de clés SSH, pas de mot de passe)  
**Optimisations :** Aucune nécessaire  
**Erreurs :** Aucune

---

### 4.8 Déploiement SSH / Docker Compose

**Statut :** ❌ Échec  
**Durée :** ~1 minute 30 secondes (jusqu'à l'échec)  
**Healthchecks :** ✅ Implémentés (PostgreSQL)  
**Timeout :** 120 secondes (60 tentatives × 2 secondes)  
**Migrations :** ❌ Échec (RuntimeError SQLALCHEMY_DATABASE_URI)  
**Redémarrage services :** ✅ Fonctionne  
**Pièges potentiels :**
- Variables d'environnement non propagées au conteneur
- Script SSH avec `set -o errexit` (bon, mais nécessite gestion d'erreurs robuste)
- Pas de rollback automatique en cas d'échec

**Correctifs :**

**Patch Script SSH - Amélioration robustesse :**

```bash
#!/bin/bash
set -o errexit -o nounset -o pipefail -x

# Fonction de rollback
rollback() {
  echo "🔄 Rollback en cours..."
  docker compose -f docker-compose.production.yml down --remove-orphans || true
  # Restaurer l'image précédente si nécessaire
  exit 1
}

trap rollback ERR

# ... reste du script ...
```

**Patch docker-compose.production.yml - Healthcheck backend :**

```yaml
services:
  backend:
    # ... autres configurations ...
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## 5. Causes Racines (Root Cause Analysis)

| Problème | Cause Racine | Impact | Priorité |
|----------|--------------|--------|----------|
| Échec migrations Flask | Variables d'environnement non propagées au conteneur Docker lors de `docker compose exec` | Blocage déploiement | 🔴 Critique |
| Conflit cache QEMU | Concurrence entre jobs GitHub Actions tentant de créer le même cache | Performance dégradée | 🟠 Haute |
| Build Docker lent | Absence de cache GitHub Actions et Docker registry | Temps de build élevé | 🟡 Moyenne |
| Warnings Trivy SBOM | Utilisation d'un SBOM tiers au lieu de Trivy | Détection imprécise | 🟠 Haute |
| Warnings pip root | Exécution pip en tant que root dans le Dockerfile | Risque permissions | 🟡 Moyenne |

---

## 6. Correctifs Actionnables

### 6.1 Patch GitHub Actions - Workflow Complet

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

env:
  DOCKER_IMAGE: your-registry/atmr-backend
  DOCKER_TAG: ${{ github.sha }}

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
        with:
          image: docker.io/tonistiigi/binfmt:latest
          platforms: all
          cache-image: true
        continue-on-error: true  # Ne pas faire échouer si cache échoue

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        with:
          driver: docker-container
          cache-binary: true

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build & Push backend image
        uses: docker/build-push-action@v5
        with:
          context: ./backend
          file: ./backend/Dockerfile.production
          build-args: WITH_RL=false
          push: true
          tags: ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
          # OPTIMISATION : Cache GitHub Actions
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image with Trivy
        uses: aquasecurity/trivy-action@master
        with:
          version: v0.67.2
          image-ref: ${{ env.DOCKER_IMAGE }}:${{ env.DOCKER_TAG }}
          format: sarif
          output: trivy-results.sarif
          ignore-unfixed: true
          vuln-type: os,library
          scan-type: image
          severity: CRITICAL,HIGH
          cache-dir: ${{ runner.temp }}/.cache/trivy

      - name: Copy compose files to server
        uses: appleboy/scp-action@v0.1.7
        with:
          host: ${{ secrets.SSH_HOST }}
          username: ${{ secrets.SSH_USER }}
          key: ${{ secrets.SSH_KEY }}
          port: ${{ secrets.SSH_PORT }}
          source: docker-compose.production.yml
          target: /srv/atmr

      - name: Deploy via SSH
        uses: appleboy/ssh-action@v0.1.10
        with:
          host: ${{ secrets.SSH_HOST }}
          username: ${{ secrets.SSH_USER }}
          key: ${{ secrets.SSH_KEY }}
          port: ${{ secrets.SSH_PORT }}
          envs: APP_ENCRYPTION_KEY_B64,SECRET_KEY,JWT_SECRET_KEY,POSTGRES_PASSWORD,POSTGRES_USER,POSTGRES_DB,MAIL_PASSWORD,SENTRY_DSN,DOCKER_IMAGE,DOCKER_TAG
          script: |
            cd /srv/atmr
            set -o errexit -o nounset -o pipefail -x
            
            # Export des variables
            export APP_ENCRYPTION_KEY_B64="${{ env.APP_ENCRYPTION_KEY_B64 }}"
            export SECRET_KEY="${{ env.SECRET_KEY }}"
            export JWT_SECRET_KEY="${{ env.JWT_SECRET_KEY }}"
            export POSTGRES_PASSWORD="${{ env.POSTGRES_PASSWORD }}"
            export POSTGRES_USER="${{ env.POSTGRES_USER }}"
            export POSTGRES_DB="${{ env.POSTGRES_DB }}"
            export MAIL_PASSWORD="${{ env.MAIL_PASSWORD }}"
            export SENTRY_DSN="${{ env.SENTRY_DSN }}"
            export DOCKER_IMAGE="${{ env.DOCKER_IMAGE }}"
            export DOCKER_TAG="${{ env.DOCKER_TAG }}"
            
            # Construction de SQLALCHEMY_DATABASE_URI
            export SQLALCHEMY_DATABASE_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
            
            echo "🔄 Pull de l'image Docker..."
            docker compose -f docker-compose.production.yml pull
            
            echo "🔄 Arrêt des conteneurs existants..."
            docker compose -f docker-compose.production.yml down --remove-orphans || true
            
            echo "🔄 Démarrage des services..."
            docker compose -f docker-compose.production.yml up -d
            
            echo "⏳ Attente que PostgreSQL soit prêt..."
            for i in $(seq 1 60); do
              POSTGRES_STATUS=$(docker compose -f docker-compose.production.yml ps postgres --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
              if [ "$POSTGRES_STATUS" = "running" ]; then
                HEALTH=$(docker inspect --format='{{.State.Health.Status}}' atmr-postgres 2>/dev/null || echo "none")
                if [ "$HEALTH" = "healthy" ]; then
                  if docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > /dev/null 2>&1; then
                    echo "✅ PostgreSQL est prêt et healthy"
                    break
                  fi
                fi
              fi
              if [ $i -eq 60 ]; then
                echo "❌ Timeout: PostgreSQL n'est pas prêt après 120 secondes"
                exit 1
              fi
              echo "  Tentative $i/60..."
              sleep 2
            done
            
            echo "🔄 Exécution des migrations de base de données..."
            # CORRECTION : Passer les variables explicitement
            docker compose -f docker-compose.production.yml exec -T \
              -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI}" \
              -e POSTGRES_USER="${POSTGRES_USER}" \
              -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
              -e POSTGRES_DB="${POSTGRES_DB}" \
              -e APP_ENCRYPTION_KEY_B64="${APP_ENCRYPTION_KEY_B64}" \
              -e SECRET_KEY="${SECRET_KEY}" \
              -e JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
              -e MAIL_PASSWORD="${MAIL_PASSWORD}" \
              -e SENTRY_DSN="${SENTRY_DSN}" \
              backend flask db upgrade || {
              echo "❌ Erreur lors de l'exécution des migrations"
              exit 1
            }
            echo "✅ Migrations appliquées avec succès"
            echo "✅ Déploiement terminé"
```

### 6.2 Patch Dockerfile.production

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Installer les dépendances système nécessaires pour la compilation
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libpq5 \
    # ... autres dépendances ...
    && apt-get autoremove -y && apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copier les requirements
COPY requirements*.txt ./

# Créer le répertoire wheels
RUN mkdir -p /wheels

# Installer les dépendances Python et créer les wheels
RUN python -m pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Installer les dépendances système runtime uniquement
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libpq5 \
    # ... autres dépendances runtime ...
    && apt-get autoremove -y && apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Créer un utilisateur non-privilégié
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copier les wheels depuis le builder
COPY --from=builder /wheels /wheels

# Copier les requirements
COPY --from=builder /app/requirements*.txt ./

# Installer les dépendances depuis les wheels
RUN python -m pip install --upgrade pip && \
    pip install --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels /root/.cache/pip

# Copier le code de l'application
COPY . .

# Changer vers l'utilisateur non-privilégié
USER appuser

# Exposer le port
EXPOSE 5000

# Commande par défaut
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### 6.3 Patch docker-compose.production.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    image: ${DOCKER_IMAGE}:${DOCKER_TAG}
    environment:
      # CORRECTION : Définir SQLALCHEMY_DATABASE_URI directement
      SQLALCHEMY_DATABASE_URI: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
      APP_ENCRYPTION_KEY_B64: ${APP_ENCRYPTION_KEY_B64}
      SECRET_KEY: ${SECRET_KEY}
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
      MAIL_PASSWORD: ${MAIL_PASSWORD}
      SENTRY_DSN: ${SENTRY_DSN}
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  postgres_data:
```

### 6.4 Patch Trivy Ignore (si nécessaire)

Créer un fichier `.trivyignore` à la racine du projet :

```text
# Ignorer les vulnérabilités connues et acceptées
# Format: CVE-YYYY-NNNNN
# Exemple:
# CVE-2024-12345  # Raison : vulnérabilité acceptée, patch non disponible
```

---

## 7. Plan d'Action Priorisé (ATMR)

### Sprint 1 (24-48h) - Fixes Critiques

**Objectif :** Débloquer le déploiement

1. ✅ **Corriger les variables d'environnement dans le script SSH** (30 min)
   - Ajouter `-e SQLALCHEMY_DATABASE_URI` dans `docker compose exec`
   - Tester le déploiement

2. ✅ **Définir SQLALCHEMY_DATABASE_URI dans docker-compose.production.yml** (15 min)
   - Ajouter la variable dans la section `environment` du service backend
   - Tester le déploiement

3. ✅ **Vérifier la propagation des variables d'environnement** (15 min)
   - Tester avec `docker compose exec backend env | grep SQLALCHEMY`
   - Valider que les migrations fonctionnent

**Effort total :** ~1 heure

---

### Sprint 2 (2-5 jours) - Fixes Haute Priorité + Optimisation

**Objectif :** Améliorer la stabilité et les performances

1. ✅ **Implémenter le cache GitHub Actions pour Docker Buildx** (1h)
   - Ajouter `cache-from: type=gha` et `cache-to: type=gha,mode=max`
   - Tester le build avec cache

2. ✅ **Corriger le conflit de cache QEMU** (30 min)
   - Ajouter `continue-on-error: true` ou utiliser une clé de cache unique
   - Tester le cache QEMU

3. ✅ **Générer le SBOM avec Trivy** (1h)
   - Ajouter une étape de génération SBOM avant le build
   - Intégrer le SBOM dans l'image Docker
   - Tester le scan Trivy

4. ✅ **Ajouter un utilisateur non-privilégié dans le Dockerfile** (30 min)
   - Créer l'utilisateur `appuser`
   - Modifier le Dockerfile pour utiliser cet utilisateur
   - Tester le build et le déploiement

**Effort total :** ~3 heures

---

### Sprint 3 (1-2 semaines) - Hardening Complet et Automatisation

**Objectif :** Améliorer la robustesse et l'automatisation

1. ✅ **Implémenter le rollback automatique** (2h)
   - Ajouter une fonction de rollback dans le script SSH
   - Tester le rollback en cas d'échec

2. ✅ **Ajouter des healthchecks pour tous les services** (1h)
   - Implémenter les healthchecks backend, celery, etc.
   - Tester les healthchecks

3. ✅ **Optimiser le Dockerfile avec BuildKit cache mounts** (2h)
   - Utiliser `--mount=type=cache` pour pip
   - Optimiser les layers Docker
   - Tester le build optimisé

4. ✅ **Implémenter le multi-arch build** (3h)
   - Configurer le build pour linux/amd64 et linux/arm64
   - Tester le build multi-arch

5. ✅ **Ajouter des tests de smoke après déploiement** (2h)
   - Implémenter des tests API basiques
   - Intégrer dans le pipeline
   - Tester les smoke tests

6. ✅ **Documenter le pipeline CI/CD** (2h)
   - Créer une documentation complète
   - Ajouter des diagrammes de flux
   - Documenter les procédures de rollback

**Effort total :** ~12 heures

---

## 8. Estimation des Efforts

| Tâche | Priorité | Effort | Complexité |
|-------|----------|--------|------------|
| Corriger variables d'environnement SSH | 🔴 Critique | 30 min | Faible |
| Définir SQLALCHEMY_DATABASE_URI dans compose | 🔴 Critique | 15 min | Faible |
| Cache GitHub Actions Docker Buildx | 🟠 Haute | 1h | Moyenne |
| Corriger conflit cache QEMU | 🟠 Haute | 30 min | Faible |
| Générer SBOM avec Trivy | 🟠 Haute | 1h | Moyenne |
| Utilisateur non-privilégié Dockerfile | 🟡 Moyenne | 30 min | Faible |
| Rollback automatique | 🟡 Moyenne | 2h | Moyenne |
| Healthchecks tous services | 🟡 Moyenne | 1h | Faible |
| Optimiser Dockerfile BuildKit | 🟡 Moyenne | 2h | Moyenne |
| Multi-arch build | 🟢 Basse | 3h | Élevée |
| Tests smoke après déploiement | 🟢 Basse | 2h | Moyenne |
| Documentation pipeline | 🟢 Basse | 2h | Faible |

**Total Sprint 1 :** ~1 heure  
**Total Sprint 2 :** ~3 heures  
**Total Sprint 3 :** ~12 heures  
**Total général :** ~16 heures (2 jours de travail)

---

## 9. Score Final du Pipeline

### Calcul du Score

| Critère | Poids | Score | Note |
|---------|-------|-------|------|
| Fonctionnalité (déploiement réussi) | 40% | 0/100 | 0 |
| Sécurité (scan Trivy) | 20% | 80/100 | 16 |
| Performance (temps de build) | 15% | 60/100 | 9 |
| Robustesse (gestion d'erreurs) | 15% | 50/100 | 7.5 |
| Maintenabilité (documentation) | 10% | 70/100 | 7 |

**Score total :** **39.5/100** ⚠️

### Justification

- **Fonctionnalité (0/100) :** Le déploiement échoue systématiquement à cause des migrations Flask. **Blocage critique.**
- **Sécurité (80/100) :** Trivy est intégré et fonctionne, mais des warnings sur le SBOM tiers réduisent la confiance.
- **Performance (60/100) :** Le build prend ~3 minutes, mais pourrait être optimisé avec le cache.
- **Robustesse (50/100) :** Pas de rollback automatique, gestion d'erreurs basique.
- **Maintenabilité (70/100) :** Le pipeline est structuré, mais manque de documentation.

### Amélioration Attendue

Après application des correctifs du Sprint 1 et Sprint 2 :
- **Score attendu :** **75/100** ✅
- **Amélioration :** +35.5 points

---

## 10. Conclusion Professionnelle

Le pipeline CI/CD ATMR présente une architecture solide avec des outils modernes (Docker Buildx, Trivy, GitHub Actions), mais souffre d'un **blocage critique** qui empêche tout déploiement réussi. La cause principale est une **mauvaise propagation des variables d'environnement** au conteneur backend lors de l'exécution des migrations Flask.

**Actions immédiates requises :**
1. Corriger la propagation des variables d'environnement dans le script SSH (30 min)
2. Définir `SQLALCHEMY_DATABASE_URI` directement dans `docker-compose.production.yml` (15 min)
3. Tester le déploiement complet (15 min)

**Recommandations stratégiques :**
- Implémenter le cache GitHub Actions pour réduire le temps de build de ~3 minutes à ~1 minute
- Générer le SBOM avec Trivy pour améliorer la précision des scans sécurité
- Ajouter un rollback automatique pour améliorer la robustesse du déploiement
- Documenter le pipeline pour faciliter la maintenance

**Risque résiduel :** Faible après application des correctifs du Sprint 1. Le pipeline sera fonctionnel et prêt pour la production.

**Prochaines étapes :**
1. Appliquer les correctifs du Sprint 1 (1 heure)
2. Valider le déploiement en environnement de test
3. Planifier le Sprint 2 pour les optimisations

---

**Rapport généré le :** 2025-11-22  
**Analyste :** Expert DevSecOps Senior  
**Version :** 1.0

