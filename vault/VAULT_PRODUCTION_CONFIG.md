# ✅ 4.1: Configuration Production HashiCorp Vault

## Vue d'ensemble

Guide pour configurer Vault en production avec haute disponibilité, audit logs, et sécurité renforcée.

## Architecture Production

```
┌─────────────────┐      ┌─────────────────┐
│   Vault Node 1  │──────│   Vault Node 2  │  (Mode HA)
│   (Active)      │      │   (Standby)     │
└────────┬────────┘      └────────┬────────┘
         │                         │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────┐
         │  Storage Backend     │  (Consul/S3/etc)
         │  (Shared State)      │
         └──────────────────────┘
```

## Étape 1 : Installation Production

### Option A : Kubernetes

```yaml
# vault/k8s/vault-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vault
spec:
  serviceName: vault
  replicas: 3 # HA avec 3 nœuds
  template:
    spec:
      containers:
        - name: vault
          image: hashicorp/vault:latest
          env:
            - name: VAULT_ADDR
              value: "https://vault.example.com"
            - name: VAULT_CLUSTER_ADDR
              value: "https://vault.example.com"
          volumeMounts:
            - name: vault-config
              mountPath: /vault/config
  volumeClaimTemplates:
    - metadata:
        name: vault-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

### Option B : Docker Compose Production

Créer `vault/docker-compose.production.yml` :

```yaml
version: "3.8"

services:
  vault:
    image: hashicorp/vault:latest
    container_name: atmr_vault_prod
    environment:
      VAULT_ADDR: "https://vault.example.com"
    volumes:
      - ./vault/config/production.hcl:/vault/config/vault.hcl:ro
      - vault_data:/vault/file
    ports:
      - "8200:8200"
    restart: always
    cap_add:
      - IPC_LOCK
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M

volumes:
  vault_data:
    driver: local
```

## Étape 2 : Configuration Production

### Fichier de configuration `vault/config/production.hcl`

```hcl
# ✅ 4.1: Configuration Vault Production

# Storage backend (Consul recommandé pour HA)
storage "consul" {
  address = "consul:8500"
  path    = "vault/"
  scheme  = "http"
  # Alternative: S3 pour cloud
  # storage "s3" {
  #   bucket = "vault-backend"
  #   region = "eu-central-1"
  # }
}

# Listener avec TLS
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/vault.crt"
  tls_key_file  = "/vault/tls/vault.key"
  tls_min_version = "1.2"
}

# API address (public URL)
api_addr = "https://vault.example.com"
cluster_addr = "https://vault.internal:8201"

# UI
ui = true

# ✅ Audit logs (OBLIGATOIRE en production)
audit_device "file" {
  path = "file"
  options = {
    file_path = "/vault/audit/vault_audit.log"
    format = "json"
  }
}

# Alternative: Syslog
# audit_device "syslog" {
#   path = "syslog"
#   options = {
#     facility = "AUTH"
#     tag = "vault"
#     format = "json"
#   }
# }

# Alternative: Socket
# audit_device "socket" {
#   path = "socket"
#   options = {
#     address = "logstash:5000"
#     socket_type = "tcp"
#     format = "json"
#   }
# }

# Performance
disable_mlock = false  # Important pour sécurité
```

## Étape 3 : Initialisation et Unseal

### Initialisation

```bash
# Générer les clés d'initialisation (garder en sécurité !)
vault operator init -key-shares=5 -key-threshold=3

# Réponse:
# Unseal Key 1: abc123...
# Unseal Key 2: def456...
# Unseal Key 3: ghi789...
# Unseal Key 4: jkl012...
# Unseal Key 5: mno345...
# Initial Root Token: s.RootToken123...

# ✅ IMPORTANT: Stocker dans coffre-fort sécurisé (1Password, etc.)
```

### Déverrouillage (Unseal)

```bash
# Déverrouiller avec 3 clés (sur 5)
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# Vérifier le statut
vault status
```

### Auto-unseal (Production recommandé)

Utiliser **Auto-unseal** pour éviter l'intervention manuelle :

```bash
# Configurer avec un service cloud (AWS KMS, GCP KMS, Azure Key Vault)
vault operator init -recovery-shares=5 -recovery-threshold=3

# Configurer auto-unseal dans vault.hcl
seal "awskms" {
  region     = "eu-central-1"
  kms_key_id = "arn:aws:kms:..."
}
```

## Étape 4 : Configuration AppRole

### Créer le rôle pour l'application ATMR

```bash
# Activer AppRole
vault auth enable approle

# Créer le rôle
vault write auth/approle/role/atmr-api \
    token_policies="atmr-api-rotate" \
    token_ttl=24h \
    token_max_ttl=48h \
    secret_id_ttl=0  # Pas d'expiration (géré par rotation)

# Obtenir le role_id
ROLE_ID=$(vault read -field=role_id auth/approle/role/atmr-api/role-id)
echo "VAULT_ROLE_ID=$ROLE_ID"

# Générer un secret_id (à stocker en Kubernetes Secret ou système de secrets)
SECRET_ID=$(vault write -f -field=secret_id auth/approle/role/atmr-api/secret-id)
echo "VAULT_SECRET_ID=$SECRET_ID"
```

### Stocker dans Kubernetes

```yaml
# vault/k8s/vault-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: vault-credentials
  namespace: atmr
type: Opaque
stringData:
  vault-role-id: "<ROLE_ID>"
  vault-secret-id: "<SECRET_ID>"
```

## Étape 5 : Variables d'Environnement Application

### Kubernetes Deployment

```yaml
# backend/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: atmr-api
spec:
  template:
    spec:
      containers:
        - name: api
          env:
            - name: VAULT_ADDR
              value: "https://vault.example.com"
            - name: VAULT_ROLE_ID
              valueFrom:
                secretKeyRef:
                  name: vault-credentials
                  key: vault-role-id
            - name: VAULT_SECRET_ID
              valueFrom:
                secretKeyRef:
                  name: vault-credentials
                  key: vault-secret-id
            - name: USE_VAULT
              value: "true"
            - name: FLASK_ENV
              value: "production"
```

### Docker Compose Production

```yaml
# docker-compose.production.yml
services:
  api:
    environment:
      VAULT_ADDR: "https://vault.example.com"
      VAULT_ROLE_ID: "${VAULT_ROLE_ID}"
      VAULT_SECRET_ID: "${VAULT_SECRET_ID}"
      USE_VAULT: "true"
      FLASK_ENV: "production"
    env_file:
      - vault-secrets.env # ⚠️ NE PAS COMMITER !
```

## Étape 6 : Configuration Audit Logs

### Activer l'audit device

```bash
# File audit (local)
vault audit enable file file_path=/vault/audit/vault_audit.log

# Syslog audit (recommandé pour centralisation)
vault audit enable syslog tag=vault facility=AUTH

# Socket audit (pour Logstash/Fluentd)
vault audit enable socket address=logstash:5000 socket_type=tcp
```

### Configuration Grafana/Prometheus

Ajouter les logs Vault à votre pipeline de logs :

```yaml
# prometheus/scrape-vault.yml
scrape_configs:
  - job_name: "vault"
    static_configs:
      - targets: ["vault:8200"]
    metrics_path: "/v1/sys/metrics"
    params:
      format: ["prometheus"]
```

## Étape 7 : Backup et Disaster Recovery

### Backup régulier

```bash
# Script de backup quotidien
#!/bin/bash
vault operator raft snapshot save /backup/vault-$(date +%Y%m%d).snapshot

# Upload vers S3
aws s3 cp /backup/vault-$(date +%Y%m%d).snapshot s3://vault-backups/
```

### Restauration

```bash
# Restaurer depuis un snapshot
vault operator raft snapshot restore /backup/vault-20240101.snapshot
```

## Étape 8 : Monitoring

### Health Checks

```bash
# Endpoint santé
curl https://vault.example.com/v1/sys/health

# Réponse:
# {
#   "initialized": true,
#   "sealed": false,
#   "standby": false,
#   "version": "1.15.0"
# }
```

### Alertes Prometheus

```yaml
# prometheus/alerts-vault.yml
groups:
  - name: vault
    rules:
      - alert: VaultSealed
        expr: vault_core_sealed == 1
        annotations:
          summary: "Vault est sealed (indisponible)"

      - alert: VaultUnsealKeyUsage
        expr: increase(vault_core_unseal_attempts_total[5m]) > 0
        annotations:
          summary: "Tentative de déverrouillage Vault détectée"

      - alert: VaultHighErrorRate
        expr: rate(vault_core_http_error_total[5m]) > 0.1
        annotations:
          summary: "Taux d'erreur Vault élevé"
```

## Checklist Production

- [ ] Vault en mode HA (3+ nœuds)
- [ ] Storage backend partagé (Consul/S3)
- [ ] TLS activé avec certificats valides
- [ ] Auto-unseal configuré (AWS KMS/GCP KMS)
- [ ] Audit logs activés (fichier/syslog/socket)
- [ ] Policies créées et testées
- [ ] AppRole configuré avec credentials sécurisés
- [ ] Backup quotidien configuré
- [ ] Monitoring et alertes actifs
- [ ] Document de disaster recovery créé
- [ ] Tests de failover effectués
- [ ] Secrets migrés depuis .env
- [ ] Rotation automatique configurée (Celery)
- [ ] Équipe formée sur Vault

## Sécurité

### Bonnes pratiques

- ✅ **Jamais** stocker root token en clair
- ✅ Utiliser auto-unseal en production
- ✅ Roter les secret_id AppRole régulièrement
- ✅ Limiter les policies (principe du moindre privilège)
- ✅ Activer les audit logs sur tous les paths
- ✅ Chiffrer les backups
- ✅ Isoler réseau (VPC privé)
- ✅ Utiliser TLS partout
- ✅ Limiter l'accès réseau (firewall)

## Troubleshooting Production

### Vault sealed

```bash
# Vérifier le statut
vault status

# Si sealed, unseal avec les clés
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>
```

### Erreur d'authentification

```bash
# Vérifier le token AppRole
vault auth -method=approle role_id=<ROLE_ID> secret_id=<SECRET_ID>

# Vérifier les policies
vault token lookup
vault policy read atmr-api-rotate
```

### Performance

```bash
# Vérifier les métriques
vault operator metrics

# Optimiser la configuration si nécessaire
```

## Ressources

- Documentation officielle : https://www.vaultproject.io/docs
- Best practices : https://learn.hashicorp.com/tutorials/vault/production-hardening
- Disaster recovery : https://www.vaultproject.io/docs/concepts/disaster-recovery
