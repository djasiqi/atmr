# 🔐 Guide GitHub Secrets - Configuration CI/CD

**Date** : 2026-01-08  
**Workflow** : `.github/workflows/deploy.yml`  
**Statut** : ✅ Workflow activé et opérationnel

---

## 🎯 RÉSUMÉ

Ce guide documente **tous les secrets GitHub** requis pour le workflow Build & Deploy automatique.

**Configuration** : GitHub → Settings → Secrets and variables → Actions → New repository secret

---

## 🔥 SECRETS CRITIQUES (OBLIGATOIRES)

Ces secrets **doivent être configurés** avant le premier déploiement, sinon le workflow échouera.

### 1. Docker Hub

| Secret               | Description                   | Exemple                  |
| -------------------- | ----------------------------- | ------------------------ |
| `DOCKER_IMAGE`       | Nom de l'image sur Docker Hub | `djasiqi/atmr-backend`   |
| `DOCKER_TAG`         | Tag de l'image                | `latest` ou `v1.0.0`     |
| `DOCKERHUB_USERNAME` | Username Docker Hub           | `djasiqi`                |
| `DOCKERHUB_TOKEN`    | Token d'accès Docker Hub      | `dckr_pat_xxxxxxxxxxxxx` |

**🔧 Comment obtenir le token Docker Hub** :

```bash
# 1. Allez sur https://hub.docker.com/settings/security
# 2. Cliquez sur "New Access Token"
# 3. Nom : "GitHub Actions ATMR"
# 4. Permissions : "Read, Write, Delete"
# 5. Copiez le token généré (commence par dckr_pat_)
```

---

### 2. SSH (Serveur de production)

| Secret     | Description                      | Exemple           |
| ---------- | -------------------------------- | ----------------- |
| `SSH_HOST` | Adresse IP ou domaine du serveur | `138.201.155.201` |
| `SSH_USER` | Utilisateur SSH                  | `deploy`          |
| `SSH_PORT` | Port SSH                         | `22`              |
| `SSH_KEY`  | Clé privée SSH (format PEM)      | (voir ci-dessous) |

**🔧 Comment obtenir la clé SSH** :

```bash
# Sur votre machine locale
cat ~/.ssh/id_rsa  # OU ~/.ssh/id_ed25519

# Copiez TOUT le contenu, y compris :
# -----BEGIN OPENSSH PRIVATE KEY-----
# ... (contenu de la clé)
# -----END OPENSSH PRIVATE KEY-----
```

**⚠️ IMPORTANT** : Ne jamais partager cette clé publiquement !

---

### 3. Base de données PostgreSQL

| Secret              | Description               | Exemple           |
| ------------------- | ------------------------- | ----------------- |
| `POSTGRES_USER`     | Utilisateur PostgreSQL    | `atmr_user`       |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL   | `***************` |
| `POSTGRES_DB`       | Nom de la base de données | `atmr_db`         |

**🔧 Comment générer un mot de passe sécurisé** :

```bash
# Générer un mot de passe de 32 caractères
openssl rand -base64 32

# OU avec Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

### 4. Redis

| Secret           | Description        | Exemple           |
| ---------------- | ------------------ | ----------------- |
| `REDIS_PASSWORD` | Mot de passe Redis | `***************` |

---

### 5. Application (Flask)

| Secret                   | Description                               | Exemple           |
| ------------------------ | ----------------------------------------- | ----------------- |
| `SECRET_KEY`             | Clé secrète Flask (sessions, CSRF)        | `***************` |
| `JWT_SECRET_KEY`         | Clé secrète JWT (authentification)        | `***************` |
| `APP_ENCRYPTION_KEY_B64` | Clé d'encryption (base64)                 | `***************` |
| `MASTER_ENCRYPTION_KEY`  | Clé maître pour IBAN et données sensibles | `***************` |

**🔧 Comment générer ces clés** :

```bash
# SECRET_KEY (Flask)
python3 -c "import secrets; print(secrets.token_hex(32))"

# JWT_SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# APP_ENCRYPTION_KEY_B64 (Fernet key)
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# MASTER_ENCRYPTION_KEY (AES-256 hex, 64 caractères)
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

### 6. CORS (Sécurité)

| Secret                  | Description                        | Exemple                                 |
| ----------------------- | ---------------------------------- | --------------------------------------- |
| `SOCKETIO_CORS_ORIGINS` | Origines autorisées pour Socket.IO | `https://www.lirie.ch,https://lirie.ch` |

**Format** : Liste séparée par des virgules (sans espaces)

---

## 🟡 SECRETS OPTIONNELS (Recommandés)

Ces secrets sont **optionnels** mais recommandés pour une configuration complète.

### 7. Email (SMTP)

| Secret          | Description                    | Exemple                |
| --------------- | ------------------------------ | ---------------------- |
| `MAIL_PASSWORD` | Mot de passe compte email SMTP | `***************`      |
| `SMTP_HOST`     | Serveur SMTP                   | `smtp.gmail.com`       |
| `SMTP_PORT`     | Port SMTP                      | `587`                  |
| `SMTP_USERNAME` | Username SMTP                  | `your-email@gmail.com` |

---

### 8. Monitoring (Grafana + Alertmanager)

| Secret                    | Description                    | Exemple                                |
| ------------------------- | ------------------------------ | -------------------------------------- |
| `GRAFANA_ADMIN_USER`      | Username admin Grafana         | `admin`                                |
| `GRAFANA_ADMIN_PASSWORD`  | Mot de passe admin Grafana     | `***************`                      |
| `GRAFANA_ROOT_URL`        | URL publique de Grafana        | `https://grafana.lirie.ch`             |
| `SLACK_WEBHOOK_URL`       | URL webhook Slack (alertes)    | `https://hooks.slack.com/services/...` |
| `ALERTMANAGER_FROM_EMAIL` | Email expéditeur des alertes   | `alerts@lirie.ch`                      |
| `ALERT_EMAIL_TO`          | Email destinataire des alertes | `admin@lirie.ch`                       |

---

### 9. Services externes

| Secret                | Description                      | Exemple                         |
| --------------------- | -------------------------------- | ------------------------------- |
| `GOOGLE_MAPS_API_KEY` | Clé API Google Maps              | `AIzaSyXXXXXXXXXXXXXXXXXXXXXX`  |
| `SENTRY_DSN`          | DSN Sentry (monitoring erreurs)  | `https://xxx@sentry.io/xxxxxxx` |
| `PDF_BASE_URL`        | URL publique pour génération PDF | `https://www.lirie.ch`          |
| `DEMO_MAGIC_LINK_BASE_URL` | URL de base pour les liens magic link demo (emails) | `https://www.lirie.ch/demo-access/consume` ou `https://demo.lirie.ch/demo-access/consume` |

**Note Demo** : Si la demo prod est sur un domaine différent (ex. `demo.lirie.ch`), définir `DEMO_MAGIC_LINK_BASE_URL` explicitement. Sinon le fallback utilise `https://www.lirie.ch/demo-access/consume` ou `http://localhost:3000` selon l'environnement.

---

## 🟢 SECRETS IGNORÉS (Non utilisés actuellement)

Ces secrets sont définis dans le workflow mais **NON utilisés** actuellement.

### 10. Grafana OAuth Google (Désactivé)

| Secret                                 | Description                           |
| -------------------------------------- | ------------------------------------- |
| `GRAFANA_OAUTH_GOOGLE_ENABLED`         | Activer OAuth Google (`true`/`false`) |
| `GRAFANA_OAUTH_GOOGLE_CLIENT_ID`       | Client ID Google OAuth                |
| `GRAFANA_OAUTH_GOOGLE_CLIENT_SECRET`   | Client Secret Google OAuth            |
| `GRAFANA_OAUTH_GOOGLE_ALLOWED_DOMAINS` | Domaines autorisés (ex: `lirie.ch`)   |

**ℹ️ Pour activer plus tard** : Décommenter les lignes correspondantes dans `.github/workflows/deploy.yml`

---

## 📋 CHECKLIST DE CONFIGURATION

Cochez les secrets que vous avez configurés :

### Docker Hub

- [ ] `DOCKER_IMAGE`
- [ ] `DOCKER_TAG`
- [ ] `DOCKERHUB_USERNAME`
- [ ] `DOCKERHUB_TOKEN`

### SSH

- [ ] `SSH_HOST`
- [ ] `SSH_USER`
- [ ] `SSH_PORT`
- [ ] `SSH_KEY`

### Base de données

- [ ] `POSTGRES_USER`
- [ ] `POSTGRES_PASSWORD`
- [ ] `POSTGRES_DB`
- [ ] `REDIS_PASSWORD`

### Application

- [ ] `SECRET_KEY`
- [ ] `JWT_SECRET_KEY`
- [ ] `APP_ENCRYPTION_KEY_B64`
- [ ] `MASTER_ENCRYPTION_KEY`
- [ ] `SOCKETIO_CORS_ORIGINS`

### Email (optionnel)

- [ ] `MAIL_PASSWORD`
- [ ] `SMTP_HOST`
- [ ] `SMTP_PORT`
- [ ] `SMTP_USERNAME`

### Monitoring (optionnel)

- [ ] `GRAFANA_ADMIN_USER`
- [ ] `GRAFANA_ADMIN_PASSWORD`
- [ ] `GRAFANA_ROOT_URL`
- [ ] `SLACK_WEBHOOK_URL`
- [ ] `ALERTMANAGER_FROM_EMAIL`
- [ ] `ALERT_EMAIL_TO`

### Services externes (optionnel)

- [ ] `GOOGLE_MAPS_API_KEY`
- [ ] `SENTRY_DSN`
- [ ] `PDF_BASE_URL`
- [ ] `DEMO_MAGIC_LINK_BASE_URL` (recommandé si demo sur domaine dédié)

---

## 🚀 TESTER LE WORKFLOW

### Option 1 : Déclenchement automatique (sur push)

```bash
# Faire un commit et push sur main
git add .
git commit -m "test: Trigger GitHub Actions workflow"
git push origin main

# Aller sur GitHub → Actions pour voir le workflow en cours
```

### Option 2 : Déclenchement manuel

```
1. Aller sur GitHub → Actions
2. Cliquer sur "Build & Deploy" dans la liste des workflows
3. Cliquer sur "Run workflow"
4. (Optionnel) Spécifier un tag personnalisé
5. Cliquer sur "Run workflow" (vert)
```

---

## 🔍 VÉRIFIER LES SECRETS

### Méthode 1 : Via l'interface GitHub

```
1. Aller sur GitHub → Settings → Secrets and variables → Actions
2. Vérifier que tous les secrets critiques sont présents
3. Les valeurs ne sont PAS visibles (normal, c'est sécurisé)
```

### Méthode 2 : Via le workflow (validation automatique)

Le workflow `deploy.yml` inclut une validation automatique des secrets :

```yaml
- name: Validate required secrets
  run: |
    REQUIRED_SECRETS=(
      "SSH_HOST"
      "SSH_USER"
      # ... (liste complète)
    )
    # Vérifie que tous les secrets sont présents
```

Si un secret manque, le workflow échouera avec un message clair :

```
❌ Secrets manquants: SECRET_KEY, JWT_SECRET_KEY
```

---

## ⚠️ SÉCURITÉ - BONNES PRATIQUES

### ✅ À FAIRE

1. **Générer des secrets aléatoires forts** (32+ caractères)
2. **Ne jamais commit les secrets** dans le code
3. **Utiliser des secrets différents** pour dev/staging/production
4. **Renouveler les secrets** tous les 6-12 mois
5. **Limiter l'accès aux secrets** (GitHub Secrets → Environment secrets pour production)

### ❌ À NE PAS FAIRE

1. **Ne jamais** partager les secrets par email/Slack/chat
2. **Ne jamais** utiliser des mots de passe simples (`password123`, `admin`, etc.)
3. **Ne jamais** logger les secrets (même pour debug)
4. **Ne jamais** hardcoder les secrets dans le code
5. **Ne jamais** push les secrets dans Git (même en `.env`)

---

## 🛟 DÉPANNAGE

### Erreur : "Secrets manquants"

```
❌ Secrets manquants: DOCKER_IMAGE, DOCKERHUB_TOKEN
```

**Solution** :

```
1. Aller sur GitHub → Settings → Secrets and variables → Actions
2. Cliquer sur "New repository secret"
3. Ajouter le secret manquant
4. Relancer le workflow
```

---

### Erreur : "Image not found on Docker Hub"

```
❌ L'image djasiqi/atmr-backend:latest n'a pas été trouvée sur Docker Hub
```

**Cause** : L'image n'a pas été pushée correctement (credentials invalides)

**Solution** :

```
1. Vérifier DOCKERHUB_USERNAME et DOCKERHUB_TOKEN
2. Régénérer le token Docker Hub si nécessaire
3. Relancer le workflow
```

---

### Erreur : "SSH connection refused"

```
❌ ssh: connect to host 138.201.155.201 port 22: Connection refused
```

**Causes possibles** :

1. **SSH_HOST incorrect** → Vérifier l'adresse IP/domaine
2. **SSH_PORT incorrect** → Vérifier le port SSH (22 par défaut)
3. **SSH_KEY incorrecte** → Vérifier la clé privée
4. **Firewall bloque GitHub Actions** → Whitelist les IP GitHub Actions

**Solution** :

```bash
# Tester la connexion SSH manuellement
ssh -p 22 deploy@138.201.155.201

# Si ça fonctionne, le problème vient des secrets GitHub
```

---

### Erreur : "Database connection failed"

```
❌ FATAL: password authentication failed for user "atmr_user"
```

**Cause** : Mot de passe PostgreSQL incorrect

**Solution** :

```
1. Vérifier POSTGRES_PASSWORD dans GitHub Secrets
2. Vérifier que le mot de passe sur le serveur correspond
3. Relancer le workflow
```

---

## 📖 RÉFÉRENCES

- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Docker Hub Access Tokens](https://docs.docker.com/docker-hub/access-tokens/)
- [SSH Key Authentication](https://www.ssh.com/academy/ssh/key)
- [Fernet Encryption (Python)](https://cryptography.io/en/latest/fernet/)

---

## 📝 HISTORIQUE DES MODIFICATIONS

| Date       | Modification                                                    | Auteur |
| ---------- | --------------------------------------------------------------- | ------ |
| 2026-01-08 | Création du guide + réactivation du workflow                    | Cursor |
| 2026-01-08 | Ajout des secrets MASTER_ENCRYPTION_KEY + SOCKETIO_CORS_ORIGINS | Cursor |

---

**Créé le** : 2026-01-08  
**Auteur** : Configuration automatique CI/CD  
**Statut** : ✅ Prêt pour utilisation
