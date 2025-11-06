# ✅ 4.1: Guide d'Installation HashiCorp Vault

## Vue d'ensemble

Ce guide explique comment installer et configurer HashiCorp Vault pour ATMR.

## Installation Docker (Développement)

### Option 1 : Docker Compose (Recommandé pour dev)

Ajouter à `docker-compose.yml` ou utiliser `vault/docker-compose.vault.yml` :

```yaml
vault:
  image: hashicorp/vault:latest
  container_name: atmr_vault
  ports:
    - "8200:8200"
  environment:
    VAULT_DEV_ROOT_TOKEN_ID: "dev-root-token" # ⚠️ DEV UNIQUEMENT
    VAULT_ADDR: "http://0.0.0.0:8200"
  cap_add:
    - IPC_LOCK
  volumes:
    - vault_data:/vault/file
    - ./vault/policies:/vault/policies:ro
    - ./vault/config:/vault/config:ro
  command: vault server -dev -dev-listen-address="0.0.0.0:8200"
  restart: unless-stopped
  # ✅ 3.6: Limits CPU/mémoire
  deploy:
    resources:
      limits:
        cpus: "0.5"
        memory: 512M
      reservations:
        cpus: "0.25"
        memory: 256M

volumes:
  vault_data:
```

### Lancer Vault

```bash
# Avec docker-compose principal
docker-compose up -d vault

# Ou avec fichier séparé
docker-compose -f docker-compose.yml -f vault/docker-compose.vault.yml up -d vault
```

### Vérifier

```bash
# Vérifier que Vault est accessible
curl http://localhost:8200/v1/sys/health

# Se connecter avec token dev
export VAULT_ADDR="http://localhost:8200"
export VAULT_TOKEN="dev-root-token"
vault status
```

## Configuration Production

### 1. Mode HA avec Storage Backend

Pour la production, utiliser un storage backend (Consul, etcd, S3) :

```hcl
# vault/config/production.hcl
storage "consul" {
  address = "consul:8500"
  path    = "vault/"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 0
  tls_cert_file = "/vault/tls/vault.crt"
  tls_key_file  = "/vault/tls/vault.key"
}

api_addr = "https://vault.example.com"
ui = true
```

### 2. Initialisation

```bash
# Générer les clés d'initialisation (garder en sécurité !)
vault operator init -key-shares=5 -key-threshold=3

# Déverrouiller avec 3 clés (sur 5)
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>
```

### 3. Créer les Policies

```bash
# Appliquer la policy pour l'API ATMR
vault policy write atmr-api-read vault/policies/atmr-api-read.hcl
vault policy write atmr-api-rotate vault/policies/atmr-api-rotate.hcl
```

### 4. Configurer AppRole

```bash
# Activer le secret engine AppRole
vault auth enable approle

# Créer le rôle pour l'application
vault write auth/approle/role/atmr-api \
    token_policies="atmr-api-read" \
    token_ttl=24h \
    token_max_ttl=48h

# Obtenir le role_id (à stocker en secret Kubernetes/environnement)
vault read auth/approle/role/atmr-api/role-id

# Générer un secret_id (pour obtenir le token)
vault write -f auth/approle/role/atmr-api/secret-id
```

## Variables d'Environnement Application

### Développement

```bash
# .env (développement)
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=dev-root-token
USE_VAULT=true
```

### Production

```bash
# Kubernetes Secret ou système de secrets
VAULT_ADDR=https://vault.example.com
VAULT_ROLE_ID=<role-id-from-vault>
VAULT_SECRET_ID=<secret-id-from-vault>
USE_VAULT=true
```

## Activation des Secrets Engines

### KV v2 (Key-Value)

```bash
# Activer KV v2
vault secrets enable -version=2 -path=atmr kv

# Créer un secret de test
vault kv put atmr/dev/flask/secret_key value="my-dev-secret-key"
```

### Database Dynamic Secrets (PostgreSQL)

```bash
# Activer le secret engine database
vault secrets enable database

# Configurer PostgreSQL
vault write database/config/postgresql \
    plugin_name=postgresql-database-plugin \
    allowed_roles="atmr-db-role" \
    connection_url="postgresql://{{username}}:{{password}}@postgres:5432/atmr?sslmode=disable" \
    username="vault_admin" \
    password="vault_admin_password"

# Créer un rôle pour générer des credentials
vault write database/roles/atmr-db-role \
    db_name=postgresql \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
    default_ttl="7d" \
    max_ttl="30d"
```

### Redis Dynamic Secrets (si nécessaire)

```bash
# Activer Redis (via plugin custom ou KV)
# Note: Redis n'a pas de plugin natif, utiliser KV pour credentials
```

## Tests

### Test de lecture

```bash
# Lire un secret
vault kv get atmr/dev/flask/secret_key

# Via API
curl \
    --header "X-Vault-Token: $VAULT_TOKEN" \
    http://localhost:8200/v1/atmr/data/dev/flask/secret_key
```

### Test de rotation

```bash
# Générer de nouveaux credentials DB
vault read database/creds/atmr-db-role
```

## Prochaines Étapes

1. ✅ Vault installé et accessible
2. ✅ Policies créées
3. ✅ Secrets engines activés
4. ➡️ Lire `vault/VAULT_MIGRATION_PLAN.md` pour migration
5. ➡️ Lire `vault/VAULT_USAGE.md` pour utilisation quotidienne
