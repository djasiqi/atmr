# ✅ 4.1: Guide d'Utilisation HashiCorp Vault pour ATMR

## Vue d'ensemble

Ce guide explique comment utiliser Vault au quotidien pour gérer les secrets de l'application ATMR.

## Prérequis

- Vault installé et accessible (voir `vault/VAULT_SETUP.md`)
- Token Vault ou credentials AppRole
- Accès aux paths secrets selon votre rôle

## Authentification

### Développement (Token)

```bash
export VAULT_ADDR="http://localhost:8200"
export VAULT_TOKEN="dev-root-token"
```

### Production (AppRole)

```bash
export VAULT_ADDR="https://vault.example.com"
export VAULT_ROLE_ID="<role-id>"
export VAULT_SECRET_ID="<secret-id>"
```

## Structure des Secrets

Tous les secrets sont stockés sous `atmr/<environment>/<category>/<key>` :

```
atmr/
├── dev/              # Développement
│   ├── flask/
│   │   └── secret_key
│   ├── jwt/
│   │   └── secret_key
│   ├── encryption/
│   │   └── master_key
│   ├── database/
│   │   └── url
│   └── mail/
│       └── password
├── staging/          # Staging
└── prod/             # Production
```

## Opérations Courantes

### Lire un Secret

#### Via CLI Vault

```bash
# Lire un secret KV v2
vault kv get atmr/dev/flask/secret_key

# Afficher seulement la valeur
vault kv get -field=value atmr/dev/flask/secret_key
```

#### Via API

```bash
curl \
    --header "X-Vault-Token: $VAULT_TOKEN" \
    http://localhost:8200/v1/atmr/data/dev/flask/secret_key
```

#### Via Application

Les secrets sont automatiquement chargés depuis Vault lors du démarrage de l'application si `VAULT_ADDR` est configuré.

### Créer/Mettre à jour un Secret

```bash
# Créer ou mettre à jour un secret
vault kv put atmr/dev/flask/secret_key value="my-secret-value"

# Avec plusieurs champs
vault kv put atmr/dev/mail \
    host="smtp.example.com" \
    port="587" \
    username="user" \
    password="pass"
```

### Lister les Secrets

```bash
# Lister les paths sous atmr/dev
vault kv list atmr/dev

# Lister récursivement
vault kv list -recursive atmr/
```

### Supprimer un Secret

```bash
# Supprimer un secret (version actuelle)
vault kv delete atmr/dev/flask/secret_key

# Supprimer toutes les versions
vault kv destroy -versions=1,2,3 atmr/dev/flask/secret_key
```

## Secrets Dynamiques (Database)

### Générer des Credentials Database

```bash
# Générer de nouveaux credentials PostgreSQL
vault read database/creds/atmr-db-role

# Exemple de réponse:
# Key                Value
# ---                -----
# lease_id           database/creds/atmr-db-role/abcd1234
# lease_duration     7d
# lease_renewable    true
# password           A1b2C3d4E5f6
# username           v-atmr-db-role-abcd1234
```

### Utilisation dans l'Application

L'application génère automatiquement des credentials via `VaultClient.get_database_credentials()`.

## Rotation des Secrets

### Rotation Manuelle

```bash
# Générer une nouvelle clé
NEW_KEY=$(openssl rand -base64 32)

# Mettre à jour dans Vault
vault kv put atmr/prod/flask/secret_key value="$NEW_KEY"

# Redémarrer l'application pour charger la nouvelle clé
```

### Rotation Automatique (via Celery)

La rotation automatique est gérée par les tasks Celery (voir `tasks/vault_rotation_tasks.py`).

```python
# Exemple de rotation programmée
celery_app.conf.beat_schedule = {
    "vault-rotate-jwt-key": {
        "task": "tasks.vault_rotation_tasks.rotate_jwt_secret",
        "schedule": 30 * 24 * 3600,  # 30 jours
    },
}
```

## Audit et Logs

### Vérifier les Logs d'Audit

```bash
# Lister les devices d'audit
vault audit list

# Lire les logs (dépend de la configuration)
# Exemple: fichier, syslog, S3, etc.
tail -f /var/log/vault/audit.log
```

### Voir l'Historique d'un Secret

```bash
# Voir les métadonnées (versions, dates)
vault kv metadata get atmr/prod/flask/secret_key

# Lire une version spécifique
vault kv get -version=1 atmr/prod/flask/secret_key
```

## Monitoring

### Health Check

```bash
# Vérifier la santé de Vault
curl http://localhost:8200/v1/sys/health

# Réponse:
# {"initialized":true,"sealed":false,"standby":false,"version":"1.15.0",...}
```

### Métriques

Vault expose des métriques Prometheus sur `/v1/sys/metrics?format=prometheus`.

## Troubleshooting

### Erreur : "permission denied"

**Cause** : Permissions insuffisantes.

**Solution** :

1. Vérifier que votre token/AppRole a les bonnes policies
2. Vérifier le path demandé correspond à votre policy

```bash
# Vérifier les policies associées à votre token
vault token lookup

# Tester les permissions
vault policy read atmr-api-read
```

### Erreur : "connection refused"

**Cause** : Vault non accessible.

**Solution** :

1. Vérifier que Vault est démarré : `vault status`
2. Vérifier `VAULT_ADDR` : `echo $VAULT_ADDR`
3. Vérifier la connectivité réseau : `curl $VAULT_ADDR/v1/sys/health`

### Secret non trouvé en application

**Cause** : Cache ou fallback .env.

**Solution** :

1. Vérifier que le secret existe : `vault kv get atmr/dev/flask/secret_key`
2. Vérifier le path dans le code (dev vs prod)
3. Vider le cache application si nécessaire
4. Vérifier les logs : `docker logs atmr_api | grep Vault`

### Erreur : "secret engine not enabled"

**Cause** : Secret engine non activé.

**Solution** :

```bash
# Activer le secret engine
vault secrets enable -version=2 -path=atmr kv
```

## Bonnes Pratiques

### 1. Ne jamais commit de secrets

- ✅ Utiliser Vault pour tous les secrets
- ❌ Ne jamais mettre de secrets dans le code ou Git

### 2. Rotation régulière

- ✅ Rotation automatique des clés JWT (30 jours)
- ✅ Rotation automatique des clés encryption (90 jours)
- ✅ Rotation automatique des credentials DB (7 jours)

### 3. Séparation des environnements

- ✅ Utiliser des paths différents : `atmr/dev/*`, `atmr/prod/*`
- ✅ Utiliser des policies différentes selon l'environnement

### 4. Audit complet

- ✅ Activer les audit logs en production
- ✅ Surveiller les accès suspects
- ✅ Conserver les logs pour conformité

### 5. Backup

- ✅ Backup régulier des données Vault
- ✅ Sauvegarder les clés d'initialisation (si mode HA)

## Ressources

- Documentation officielle : https://www.vaultproject.io/docs
- Client Python (hvac) : https://hvac.readthedocs.io/
- Migration : `vault/VAULT_MIGRATION_PLAN.md`
- Setup : `vault/VAULT_SETUP.md`
