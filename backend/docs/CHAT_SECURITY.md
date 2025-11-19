# 🔒 Sécurité Chat - Documentation

## ✅ Fonctionnalités implémentées

### 1. **Antivirus ClamAV**

- **Service**: `backend/services/clamav_service.py`
- **Configuration**: Variables d'environnement
  - `CLAMAV_ENABLED=true` (désactivé par défaut)
  - `CLAMAV_HOST=127.0.0.1` (ou `clamav` en Docker)
  - `CLAMAV_PORT=3310`
  - `CLAMAV_TIMEOUT=5` (secondes)
- **Comportement**: Fail-open (accepte le fichier si ClamAV indisponible)
- **Intégration**: Scan automatique lors de l'upload

### 2. **Anti-spam Redis**

- **Service**: `backend/services/spam_protection.py`
- **Limite**: 1 message par seconde par utilisateur
- **Configuration**: Variables d'environnement
  - `SPAM_RATE_LIMIT_SECONDS=1.0` (1 seconde par défaut)
  - `SPAM_REDIS_TTL=2` (TTL de la clé Redis)
- **Comportement**: Fail-open (autorise si Redis indisponible)
- **Intégration**: Vérification dans Socket.IO `team_chat_message`

### 3. **Limite 1 fichier par message**

- ✅ Validation dans l'endpoint upload (`/api/v1/messages/upload`)
- ✅ Validation dans Socket.IO (image OU PDF, pas les deux)

### 4. **Validation MIME type**

- ✅ Validation par extension ET par MIME type
- ✅ Types autorisés:
  - Images: `image/jpeg`, `image/png`, `image/jpg`, `image/webp`, `image/gif`
  - PDF: `application/pdf`

### 5. **Endpoint Upload sécurisé**

- **URL**: `POST /api/v1/messages/upload`
- **Champ**: `file` (FormData)
- **Validations**:
  - ✅ Extension autorisée
  - ✅ MIME type autorisé
  - ✅ Taille max: 10 Mo
  - ✅ Limite: 1 fichier
  - ✅ Scan ClamAV
- **Retour**: `{"url": "...", "filename": "...", "size_bytes": ..., "file_type": "image"|"pdf"}`

## 📋 Configuration Docker (ClamAV)

### Option 1: ClamAV en conteneur séparé

Ajouter dans `docker-compose.yml`:

```yaml
clamav:
  image: clamav/clamav:latest
  ports:
    - "3310:3310"
  networks:
    - internal
  restart: unless-stopped
```

### Option 2: ClamAV sur le serveur

```bash
apt-get install clamav clamav-daemon -y
systemctl enable clamav-daemon
systemctl start clamav-daemon
freshclam  # Mise à jour des signatures
```

## 🔧 Variables d'environnement

```bash
# ClamAV (optionnel)
CLAMAV_ENABLED=true
CLAMAV_HOST=clamav  # ou 127.0.0.1
CLAMAV_PORT=3310
CLAMAV_TIMEOUT=5

# Anti-spam (optionnel, utilise Redis par défaut)
SPAM_RATE_LIMIT_SECONDS=1.0
SPAM_REDIS_TTL=2
```

## 📊 Résumé des sécurités

| Fonctionnalité        | Statut        | Fail-open |
| --------------------- | ------------- | --------- |
| ClamAV                | ✅ Implémenté | ✅ Oui    |
| Anti-spam             | ✅ Implémenté | ✅ Oui    |
| Limite 1 fichier      | ✅ Implémenté | ❌ Non    |
| Validation MIME       | ✅ Implémenté | ❌ Non    |
| Validation extension  | ✅ Implémenté | ❌ Non    |
| Limite taille (10 Mo) | ✅ Implémenté | ❌ Non    |

## 🚀 Migration

La migration `add_message_file_fields` sera appliquée automatiquement lors du prochain déploiement Docker via le service `migrations`.
