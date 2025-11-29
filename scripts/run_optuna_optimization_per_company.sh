#!/bin/bash
# Script pour exécuter l'optimisation Optuna par entreprise sur le serveur
# Usage: ./run_optuna_optimization_per_company.sh [COMPANY_ID]

set -euo pipefail

# Charger les variables d'environnement depuis .env.rl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ENV_FILE="${SCRIPT_DIR}/../.env.rl"

if [ -f "${RL_ENV_FILE}" ]; then
    echo "📋 Chargement des variables depuis ${RL_ENV_FILE}..."
    # shellcheck source=/dev/null
    source "${RL_ENV_FILE}"
else
    echo "⚠️  Fichier .env.rl non trouvé, utilisation des variables d'environnement système"
fi

# Variables par défaut
RL_POSTGRES_USER="${RL_POSTGRES_USER:-atmr_rl_user}"
RL_POSTGRES_DB="${RL_POSTGRES_DB:-atmr_rl_db}"
RL_POSTGRES_HOST="${RL_POSTGRES_HOST:-rl-postgres}"
RL_POSTGRES_PORT="${RL_POSTGRES_PORT:-5432}"

POSTGRES_USER="${POSTGRES_USER:-atmr_user}"
POSTGRES_DB="${POSTGRES_DB:-atmr_db}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

# Configuration Optuna (peut être surchargée par variables d'environnement)
OPTUNA_DATA_PERIOD="${OPTUNA_DATA_PERIOD:-week}"
OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-30}"
OPTUNA_TRAINING_EPISODES="${OPTUNA_TRAINING_EPISODES:-150}"
OPTUNA_EVAL_EPISODES="${OPTUNA_EVAL_EPISODES:-15}"

# Company ID optionnel (premier argument)
if [ $# -gt 0 ]; then
    OPTUNA_COMPANY_ID="$1"
    echo "🎯 Optimisation pour l'entreprise ID: ${OPTUNA_COMPANY_ID}"
else
    OPTUNA_COMPANY_ID="${OPTUNA_COMPANY_ID:-}"
    if [ -n "${OPTUNA_COMPANY_ID}" ]; then
        echo "🎯 Optimisation pour l'entreprise ID: ${OPTUNA_COMPANY_ID}"
    else
        echo "📊 Optimisation pour toutes les entreprises"
    fi
fi

# Validation des variables critiques
if [ -z "${RL_POSTGRES_PASSWORD:-}" ]; then
    echo "❌ Erreur: RL_POSTGRES_PASSWORD n'est pas défini"
    exit 1
fi

if [ -z "${POSTGRES_PASSWORD:-}" ]; then
    echo "❌ Erreur: POSTGRES_PASSWORD n'est pas défini"
    exit 1
fi

echo "🚀 Démarrage de l'optimisation Optuna par entreprise..."
echo "📊 Configuration:"
echo "   - Période de données: ${OPTUNA_DATA_PERIOD}"
echo "   - Nombre de trials: ${OPTUNA_N_TRIALS}"
echo "   - Épisodes d'entraînement: ${OPTUNA_TRAINING_EPISODES}"
echo "   - Épisodes d'évaluation: ${OPTUNA_EVAL_EPISODES}"
echo ""

# Exécuter dans un conteneur Docker temporaire
docker run --rm \
  --network atmr-rl-network \
  -e RL_POSTGRES_USER="${RL_POSTGRES_USER}" \
  -e RL_POSTGRES_PASSWORD="${RL_POSTGRES_PASSWORD}" \
  -e RL_POSTGRES_DB="${RL_POSTGRES_DB}" \
  -e RL_POSTGRES_HOST="${RL_POSTGRES_HOST}" \
  -e RL_POSTGRES_PORT="${RL_POSTGRES_PORT}" \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -e POSTGRES_DB="${POSTGRES_DB}" \
  -e POSTGRES_HOST="${POSTGRES_HOST}" \
  -e POSTGRES_PORT="${POSTGRES_PORT}" \
  -e OPTUNA_DATA_PERIOD="${OPTUNA_DATA_PERIOD}" \
  -e OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS}" \
  -e OPTUNA_TRAINING_EPISODES="${OPTUNA_TRAINING_EPISODES}" \
  -e OPTUNA_EVAL_EPISODES="${OPTUNA_EVAL_EPISODES}" \
  ${OPTUNA_COMPANY_ID:+-e OPTUNA_COMPANY_ID="${OPTUNA_COMPANY_ID}"} \
  python:3.11-slim \
  sh -c "
    set -euo pipefail
    echo '📦 Installation des dépendances...'
    apt-get update -qq
    apt-get install -y -qq git > /dev/null 2>&1
    pip install -q optuna psycopg2-binary sqlalchemy pandas gymnasium torch > /dev/null 2>&1
    echo '📥 Clonage du dépôt Git...'
    git clone --depth 1 https://github.com/djasiqi/atmr.git /tmp/atmr > /dev/null 2>&1
    cd /tmp/atmr
    echo '🚀 Exécution du script d'optimisation...'
    python3 scripts/run_optuna_optimization_per_company.py
  "

echo ""
echo "✅ Optimisation terminée !"
echo "🌐 Accédez au dashboard: https://optuna.lirie.ch"

