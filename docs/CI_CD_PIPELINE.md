# Documentation du Pipeline CI/CD - ATMR

## Vue d'ensemble

Ce document décrit l'architecture, la configuration et les procédures du pipeline CI/CD pour l'application ATMR.

## Architecture du Pipeline

Le pipeline CI/CD est composé de plusieurs étapes :

1. **Checkout** : Récupération du code source
2. **Validation des secrets** : Vérification que tous les secrets requis sont présents
3. **Setup QEMU** : Configuration pour le build multi-architecture (avec gestion d'erreur)
4. **Setup Docker Buildx** : Configuration de Docker Buildx pour les builds optimisés
5. **Génération SBOM** : Génération du Software Bill of Materials avec Trivy
6. **Login Docker Hub** : Authentification auprès de Docker Hub
7. **Build & Push** : Construction et push de l'image Docker avec cache GitHub Actions
8. **Scan Trivy** : Scan de sécurité de l'image Docker
9. **Copy files** : Copie des fichiers de configuration vers le serveur
10. **Deploy via SSH** : Déploiement automatique sur le serveur de production

## Variables d'environnement requises

### Secrets GitHub Actions

Les secrets suivants doivent être configurés dans GitHub Actions :

#### Secrets de déploiement

- `DOCKER_IMAGE` : Nom de l'image Docker (ex: `docker.io/djasiqi/atmr-backend`)
- `DOCKER_TAG` : Tag de l'image Docker (ex: `latest`, `v1.0.0`)
- `SSH_HOST` : Adresse IP ou hostname du serveur de production
- `SSH_USER` : Utilisateur SSH pour le déploiement
- `SSH_KEY` : Clé privée SSH pour l'authentification
- `SSH_PORT` : Port SSH (généralement `22`)

#### Secrets applicatifs

- `APP_ENCRYPTION_KEY_B64` : Clé de chiffrement de l'application (base64)
- `SECRET_KEY` : Clé secrète Flask
- `JWT_SECRET_KEY` : Clé secrète pour JWT
- `POSTGRES_PASSWORD` : Mot de passe PostgreSQL
- `POSTGRES_USER` : Utilisateur PostgreSQL
- `POSTGRES_DB` : Nom de la base de données PostgreSQL
- `MAIL_PASSWORD` : Mot de passe pour l'envoi d'emails (optionnel)
- `SENTRY_DSN` : DSN Sentry pour le monitoring (optionnel)
- `DOCKERHUB_USERNAME` : Nom d'utilisateur Docker Hub
- `DOCKERHUB_TOKEN` : Token d'authentification Docker Hub

### Variables d'environnement sur le serveur

Les variables suivantes sont exportées sur le serveur lors du déploiement :

- `APP_ENCRYPTION_KEY_B64`
- `SECRET_KEY`
- `JWT_SECRET_KEY`
- `POSTGRES_PASSWORD`
- `POSTGRES_USER`
- `POSTGRES_DB`
- `MAIL_PASSWORD`
- `SENTRY_DSN`
- `DOCKER_IMAGE`
- `DOCKER_TAG`

## Procédures de Rollback

### Rollback automatique

Le pipeline implémente un rollback automatique en cas d'erreur lors du déploiement. Si une étape échoue, le script de déploiement :

1. Arrête tous les conteneurs
2. Supprime les conteneurs orphelins
3. Affiche un message d'erreur
4. Quitte avec un code d'erreur

### Rollback manuel

Pour effectuer un rollback manuel :

```bash
cd /srv/atmr
docker compose -f docker-compose.production.yml down --remove-orphans
# Optionnel : restaurer une image précédente
docker pull ${DOCKER_IMAGE}:${PREVIOUS_TAG}
export DOCKER_TAG=${PREVIOUS_TAG}
docker compose -f docker-compose.production.yml up -d
```

## Troubleshooting commun

### Erreur : "SQLALCHEMY_DATABASE_URI must be set"

**Cause** : Les variables d'environnement PostgreSQL ne sont pas correctement propagées.

**Solution** :

1. Vérifier que `POSTGRES_USER`, `POSTGRES_PASSWORD`, et `POSTGRES_DB` sont définis dans les secrets GitHub
2. Vérifier que `SQLALCHEMY_DATABASE_URI` est défini dans `docker-compose.production.yml`
3. Vérifier que les variables sont exportées dans le script SSH

### Erreur : "Failed to save: Unable to reserve cache"

**Cause** : Conflit de cache QEMU lors de l'exécution simultanée de plusieurs jobs.

**Solution** : Le pipeline utilise `continue-on-error: true` pour cette étape. L'erreur est ignorée et n'affecte pas le build.

### Erreur : "PostgreSQL n'est pas prêt après 120 secondes"

**Cause** : PostgreSQL prend trop de temps à démarrer ou rencontre une erreur.

**Solution** :

1. Vérifier les logs PostgreSQL : `docker compose -f docker-compose.production.yml logs postgres`
2. Vérifier que les volumes PostgreSQL ne sont pas corrompus
3. Augmenter le timeout dans le script de déploiement si nécessaire

### Erreur : "Les smoke tests ont échoué"

**Cause** : Les services ne répondent pas correctement après le déploiement.

**Solution** :

1. Vérifier l'état des conteneurs : `docker compose -f docker-compose.production.yml ps`
2. Vérifier les logs du backend : `docker compose -f docker-compose.production.yml logs backend`
3. Vérifier que l'endpoint `/health` répond : `curl http://localhost:5000/health`

### Erreur : "Secrets manquants"

**Cause** : Un ou plusieurs secrets requis ne sont pas configurés dans GitHub Actions.

**Solution** :

1. Vérifier la liste des secrets requis dans la section "Validate required secrets" du workflow
2. Ajouter les secrets manquants dans les paramètres du dépôt GitHub
3. Vérifier que les noms des secrets correspondent exactement

## Optimisations du Pipeline

### Cache GitHub Actions

Le pipeline utilise le cache GitHub Actions pour accélérer les builds Docker :

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

### Cache BuildKit

Le Dockerfile utilise les cache mounts BuildKit pour optimiser l'installation des dépendances Python :

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel
```

### Scan de sécurité

Le pipeline effectue deux scans de sécurité :

1. **SBOM** : Génération du Software Bill of Materials avant le build
2. **Trivy** : Scan de vulnérabilités de l'image Docker (CRITICAL et HIGH uniquement)

## Healthchecks

Tous les services ont des healthchecks configurés :

- **backend** : Vérifie que l'endpoint `/health` répond avec status 200
- **postgres** : Vérifie que PostgreSQL est prêt et accessible
- **redis** : Vérifie que Redis répond aux commandes
- **celery-worker** : Vérifie que le worker Celery répond aux commandes
- **celery-beat** : Vérifie que le fichier de schedule existe

## Smoke Tests

Les smoke tests sont exécutés automatiquement après le déploiement pour valider :

1. L'endpoint `/health` répond avec status 200
2. La réponse contient `"status": "healthy"`
3. La base de données est accessible
4. Les migrations sont à jour

Le script de smoke tests se trouve dans `scripts/smoke_tests.sh`.

## Monitoring

### Logs

Les logs des services sont disponibles via :

```bash
docker compose -f docker-compose.production.yml logs [service]
```

### Métriques

Les métriques sont collectées via OpenTelemetry (configuré dans `app.py`).

## Maintenance

### Mise à jour des dépendances

1. Mettre à jour `requirements.txt` ou `requirements-rl.txt`
2. Le cache Docker sera invalidé automatiquement
3. Le build suivra avec les nouvelles dépendances

### Mise à jour de l'image de base

1. Modifier `FROM python:3.11-slim-bookworm` dans `Dockerfile.production`
2. Le cache sera invalidé et le build utilisera la nouvelle image

### Ajout de nouveaux secrets

1. Ajouter le secret dans GitHub Actions
2. Ajouter la variable dans la section `env` du workflow
3. Exporter la variable dans le script SSH si nécessaire
4. Ajouter la variable dans `docker-compose.production.yml` si nécessaire

## Support

Pour toute question ou problème, consulter :

- Le fichier `AUDIT_CI_CD_2025.md` pour les problèmes connus et solutions
- Les logs GitHub Actions pour les erreurs détaillées
- Les logs Docker sur le serveur pour les problèmes de runtime
