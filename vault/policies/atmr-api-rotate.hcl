# ✅ 4.1: Policy Vault - Lecture + Rotation pour API ATMR
# Permet à l'application de lire et de faire tourner les secrets

# Lecture des secrets
path "atmr/data/*" {
  capabilities = ["read"]
}

# Métadonnées
path "atmr/metadata/*" {
  capabilities = ["read", "list"]
}

# Rotation des clés (écriture pour nouvelle version)
path "atmr/data/*" {
  capabilities = ["read", "create", "update"]
}

# Secrets dynamiques Database
path "database/creds/atmr-db-role" {
  capabilities = ["read"]
}

# Secrets dynamiques Redis (si configuré)
path "redis/creds/atmr-redis-role" {
  capabilities = ["read"]
}

# Information système
path "sys/health" {
  capabilities = ["read", "sudo"]
}

# Rotation automatique (si configuré via API)
path "sys/rotate" {
  capabilities = ["update"]
}

