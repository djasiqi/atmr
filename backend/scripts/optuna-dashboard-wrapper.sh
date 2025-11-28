#!/bin/bash
# Script wrapper pour Optuna Dashboard qui construit l'URL PostgreSQL
# depuis les variables d'environnement de manière sécurisée

set -e

# Lire les variables d'environnement
RL_POSTGRES_USER="${RL_POSTGRES_USER:-atmr_rl_user}"
RL_POSTGRES_PASSWORD="${RL_POSTGRES_PASSWORD:-atmr_rl_password}"
RL_POSTGRES_DB="${RL_POSTGRES_DB:-atmr_rl_db}"
RL_POSTGRES_HOST="${RL_POSTGRES_HOST:-rl-postgres}"
RL_POSTGRES_PORT="${RL_POSTGRES_PORT:-5432}"

# Encoder le mot de passe pour l'URL (caractères spéciaux)
# Utiliser Python pour encoder correctement
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote_plus; print(quote_plus('${RL_POSTGRES_PASSWORD}'))")

# Construire l'URL PostgreSQL
POSTGRES_URL="postgresql://${RL_POSTGRES_USER}:${ENCODED_PASSWORD}@${RL_POSTGRES_HOST}:${RL_POSTGRES_PORT}/${RL_POSTGRES_DB}"

# Exécuter Optuna Dashboard avec l'URL construite
exec optuna-dashboard --host 0.0.0.0 --port 8080 "${POSTGRES_URL}"

