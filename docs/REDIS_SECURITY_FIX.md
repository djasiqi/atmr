# Correction de sécurité Redis - Exposition publique

## 🔴 Problème identifié

Le serveur Redis était **exposé publiquement sur Internet** sans authentification, ce qui constitue une **vulnérabilité de sécurité critique**.

**Notification reçue :**
- **Source :** BSI (Bundesamt für Sicherheit in der Informationstechnik) / CERT-Bund
- **Date :** 2025-11-25 16:11:55 UTC
- **IP affectée :** 138.201.155.201
- **Version Redis :** 7.4.7
- **Problème :** Redis accessible publiquement sans authentification SASL

## ✅ Corrections appliquées

### 1. Suppression de l'exposition publique du port Redis

**Avant :**
```yaml
ports:
  - "6379:6379"  # ❌ Exposé publiquement
```

**Après :**
```yaml
expose:
  - "6379"  # ✅ Uniquement accessible via le réseau Docker interne
```

### 2. Ajout de l'authentification Redis

**Configuration Redis :**
- Ajout de `--requirepass ${REDIS_PASSWORD}` dans la commande Redis
- Activation de `--protected-mode yes`
- Healthcheck mis à jour pour utiliser l'authentification

### 3. Mise à jour des URLs Redis dans l'application

Toutes les connexions Redis utilisent maintenant l'authentification :

**Avant :**
```yaml
CELERY_BROKER_URL: redis://redis:6379/0
CELERY_RESULT_BACKEND: redis://redis:6379/0
REDIS_URL: redis://redis:6379/0
```

**Après :**
```yaml
CELERY_BROKER_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND: redis://:${REDIS_PASSWORD}@redis:6379/0
REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
```

### 4. Ajout du secret REDIS_PASSWORD dans GitHub Actions

Le secret `REDIS_PASSWORD` a été ajouté au workflow de déploiement.

## 📋 Actions requises

### 1. Ajouter le secret REDIS_PASSWORD dans GitHub Actions

1. Allez dans votre dépôt GitHub
2. **Settings** → **Secrets and variables** → **Actions**
3. Cliquez sur **New repository secret**
4. Nom : `REDIS_PASSWORD`
5. Valeur : Générez un mot de passe fort (minimum 32 caractères recommandé)

**Génération d'un mot de passe sécurisé :**
```bash
# Option 1: OpenSSL
openssl rand -base64 32

# Option 2: Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Option 3: /dev/urandom
head -c 32 /dev/urandom | base64
```

### 2. Redéployer l'application

Après avoir ajouté le secret `REDIS_PASSWORD`, redéployez l'application :

```bash
# Via GitHub Actions
# Déclenchez manuellement le workflow "Build & Deploy"
```

### 3. Vérifier la sécurité

Après le déploiement, vérifiez que Redis n'est plus accessible publiquement :

```bash
# Depuis l'extérieur du serveur (devrait échouer)
redis-cli -h 138.201.155.201 -p 6379 ping
# ERR AUTH <password> required

# Depuis le serveur (devrait fonctionner avec le mot de passe)
docker compose -f docker-compose.production.yml exec redis redis-cli -a "${REDIS_PASSWORD}" ping
# PONG
```

### 4. Vérifier le pare-feu (recommandé)

Assurez-vous que le pare-feu bloque le port 6379 depuis l'extérieur :

```bash
# Sur le serveur
sudo ufw deny 6379/tcp
# ou
sudo iptables -A INPUT -p tcp --dport 6379 -j DROP
```

## 🔒 Sécurité renforcée

### Mesures de sécurité appliquées :

1. ✅ **Port non exposé publiquement** : Redis n'est accessible que via le réseau Docker interne
2. ✅ **Authentification requise** : Mot de passe obligatoire pour toutes les connexions
3. ✅ **Protected mode activé** : Protection supplémentaire de Redis
4. ✅ **URLs sécurisées** : Toutes les connexions incluent l'authentification

### Recommandations supplémentaires :

1. **Pare-feu** : Bloquer le port 6379 au niveau du pare-feu système
2. **VPN/Tunnel SSH** : Si vous devez accéder à Redis depuis l'extérieur, utilisez un tunnel SSH ou un VPN
3. **Rotation des mots de passe** : Changez régulièrement le mot de passe Redis
4. **Monitoring** : Surveillez les tentatives de connexion non autorisées

## 📝 Notes importantes

- **Pas de réponse nécessaire** : Vous n'avez pas besoin de répondre au BSI ou à Hetzner
- **Vérification automatique** : Le BSI vérifiera automatiquement que le problème est résolu
- **Pas de nouvelles notifications** : Vous ne devriez plus recevoir de notifications après correction

## 🔗 Références

- [Redis Security Documentation](https://redis.io/docs/management/security/)
- [BSI CERT-Bund Reports](https://reports.cert-bund.de/en/)
- [Hetzner Abuse Team](https://www.hetzner.com/abuse)

## ✅ Checklist de déploiement

- [ ] Secret `REDIS_PASSWORD` ajouté dans GitHub Actions
- [ ] Application redéployée avec les nouvelles configurations
- [ ] Vérification que Redis n'est plus accessible publiquement
- [ ] Vérification que l'application fonctionne correctement avec Redis authentifié
- [ ] Pare-feu configuré pour bloquer le port 6379 (optionnel mais recommandé)

