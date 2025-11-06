# ✅ 4.1: Policy Vault - Lecture seule pour API ATMR
# Permet à l'application de lire les secrets mais pas de les modifier

path "atmr/data/*" {
  capabilities = ["read"]
}

# Métadonnées (pour vérifier existence)
path "atmr/metadata/*" {
  capabilities = ["read", "list"]
}

# Secrets dynamiques Database
path "database/creds/atmr-db-role" {
  capabilities = ["read"]
}

# Secrets dynamiques Redis (si configuré)
path "redis/creds/atmr-redis-role" {
  capabilities = ["read"]
}

# Information système (pour health check)
path "sys/health" {
  capabilities = ["read", "sudo"]
}

# Rotation de clés (lecture seulement, pas de modification)
path "atmr/rotate/*" {
  capabilities = ["read"]
}

