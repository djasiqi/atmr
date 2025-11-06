# ✅ 4.1: Quick Start HashiCorp Vault pour ATMR

## Démarrage Rapide (5 minutes)

### 1. Installer hvac

```bash
cd backend
pip install hvac>=1.2.0
# Ou ajouter à requirements.txt et installer via pip install -r requirements.txt
```

### 2. Lancer Vault en développement

```bash
# Depuis la racine du projet
docker-compose -f docker-compose.yml -f vault/docker-compose.vault.yml up -d vault

# Vérifier que Vault est accessible
curl http://localhost:8200/v1/sys/health
```

### 3. Configurer les variables d'environnement

```bash
# backend/.env (ou variables d'environnement)
export VAULT_ADDR="http://localhost:8200"
export VAULT_TOKEN="dev-root-token"
export USE_VAULT="true"
```

### 4. Migrer les secrets depuis .env

```bash
# Depuis la racine du projet
python vault/migrate-secrets.py \
    --env-file backend/.env \
    --vault-addr http://localhost:8200 \
    --vault-token dev-root-token \
    --environment dev

# Ou en dry-run d'abord
python vault/migrate-secrets.py \
    --env-file backend/.env \
    --environment dev \
    --dry-run
```

### 5. Tester l'application

```bash
# Lancer l'application
cd backend
flask run

# Vérifier les logs : vous devriez voir "[4.1 Vault] Client Vault initialisé"
```

### 6. Vérifier la rotation automatique (optionnel)

Les tasks Celery de rotation sont automatiquement configurées dans `celery_app.py`.

Pour tester manuellement :

```bash
# Via Celery Beat (si démarré)
# Les rotations s'exécutent automatiquement selon le schedule

# Ou manuellement via Python
python -c "
from celery_app import celery
result = celery.send_task('tasks.vault_rotation_tasks.rotate_jwt_secret')
print(result.get())
"
```

## Problèmes Courants

### "Connection refused"

**Cause** : Vault n'est pas démarré.

**Solution** :

```bash
docker-compose -f docker-compose.yml -f vault/docker-compose.vault.yml up -d vault
docker logs atmr_vault
```

### "permission denied"

**Cause** : Token invalide ou policy insuffisante.

**Solution** :

```bash
# Vérifier le token
vault token lookup

# Vérifier les policies
vault policy read atmr-api-read
```

### "Secret non trouvé" en application

**Cause** : Secret pas migré ou path incorrect.

**Solution** :

```bash
# Vérifier que le secret existe
vault kv get atmr/dev/flask/secret_key

# Re-migrer si nécessaire
python vault/migrate-secrets.py --env-file backend/.env --environment dev
```

### Application utilise encore .env au lieu de Vault

**Cause** : `VAULT_ADDR` non configuré ou Vault indisponible.

**Solution** :

```bash
# Vérifier les variables
echo $VAULT_ADDR
echo $VAULT_TOKEN

# L'application fallback automatiquement vers .env si Vault indisponible
# C'est normal en développement si Vault n'est pas lancé
```

## Prochaines Étapes

1. ✅ **Développement** : Vault fonctionne en mode dev
2. ➡️ **Production** : Consulter `vault/VAULT_PRODUCTION_CONFIG.md`
3. ➡️ **Migration complète** : Suivre `vault/VAULT_MIGRATION_PLAN.md`
4. ➡️ **Utilisation quotidienne** : Consulter `vault/VAULT_USAGE.md`

## Documentation Complète

- **Setup** : `vault/VAULT_SETUP.md`
- **Usage** : `vault/VAULT_USAGE.md`
- **Migration** : `vault/VAULT_MIGRATION_PLAN.md`
- **Production** : `vault/VAULT_PRODUCTION_CONFIG.md`
