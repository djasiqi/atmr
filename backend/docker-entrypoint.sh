#!/usr/bin/env bash
# docker-entrypoint.sh
# Script d'entrée Docker avec warmup des modèles ML et vérifications de santé

set -euo pipefail

# Configuration des logs
exec > >(tee -a /app/logs/docker-entrypoint.log)
exec 2>&1

echo "🚀 Démarrage du conteneur ATMR Backend..."
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "CPU Cores: $(nproc)"

# Variables d'environnement par défaut
export FLASK_ENV="${FLASK_ENV:-production}"
export FLASK_APP="${FLASK_APP:-app.py}"
export PYTHONPATH="${PYTHONPATH:-/app}"

# Optimisations PyTorch pour CPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "🔧 Configuration:"
echo "  FLASK_ENV: $FLASK_ENV"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"

# Fonction de warmup des modèles ML
warmup_models() {
    echo "🔥 Warmup des modèles ML..."
    
    # Créer le répertoire pour les modèles s'il n'existe pas
    mkdir -p /app/data/ml /app/data/rl
    
    # Warmup du modèle de prédiction de retard
    if [ -f "/app/data/ml/delay_predictor.pkl" ]; then
        echo "  📊 Chargement du modèle de prédiction de retard..."
        python -c "
import pickle
import logging
logging.basicConfig(level=logging.INFO)
try:
    with open('/app/data/ml/delay_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f'✅ Modèle de prédiction de retard chargé: {type(model).__name__}')
except Exception as e:
    print(f'⚠️  Erreur lors du chargement du modèle de prédiction: {e}')
"
    else
        echo "  ⚠️  Modèle de prédiction de retard non trouvé"
    fi
    
    # Warmup des modèles RL
    if [ -f "/app/data/rl/best_model.pth" ]; then
        echo "  🤖 Chargement du modèle RL..."
        python -c "
import torch
import logging
logging.basicConfig(level=logging.INFO)
try:
    model = torch.load('/app/data/rl/best_model.pth', map_location='cpu')
    print(f'✅ Modèle RL chargé: {type(model).__name__}')
    # Test d'inférence pour vérifier le modèle
    if hasattr(model, 'forward'):
        dummy_input = torch.randn(1, 10)  # Exemple d'input
        with torch.no_grad():
            _ = model(dummy_input)
        print('✅ Test d\'inférence RL réussi')
except Exception as e:
    print(f'⚠️  Erreur lors du chargement du modèle RL: {e}')
"
    else
        echo "  ⚠️  Modèle RL non trouvé"
    fi
    
    # Warmup des scalers
    if [ -f "/app/data/ml/scalers.json" ]; then
        echo "  📏 Chargement des scalers..."
        python -c "
import json
import logging
logging.basicConfig(level=logging.INFO)
try:
    with open('/app/data/ml/scalers.json', 'r') as f:
        scalers = json.load(f)
    print(f'✅ Scalers chargés: {len(scalers)} scalers disponibles')
except Exception as e:
    print(f'⚠️  Erreur lors du chargement des scalers: {e}')
"
    else
        echo "  ⚠️  Scalers non trouvés"
    fi
    
    echo "✅ Warmup des modèles terminé"
}

# Fonction de vérification de la base de données
check_database() {
    echo "🗄️  Vérification de la base de données..."
    
    python -c "
import os
import logging
logging.basicConfig(level=logging.INFO)

# Vérifier les variables d'environnement de la DB
db_url = os.getenv('DATABASE_URL', '')
if db_url:
    print(f'✅ DATABASE_URL configurée: {db_url[:20]}...')
else:
    print('⚠️  DATABASE_URL non configurée')

# Test de connexion si possible
try:
    from sqlalchemy import create_engine
    if db_url:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute('SELECT 1')
            print('✅ Connexion à la base de données réussie')
except Exception as e:
    print(f'⚠️  Erreur de connexion à la base de données: {e}')
"
}

# Fonction de vérification de Redis
check_redis() {
    echo "🔴 Vérification de Redis..."
    
    python -c "
import os
import logging
logging.basicConfig(level=logging.INFO)

redis_url = os.getenv('CELERY_BROKER_URL', 'redis://127.0.0.1:6379/0')
print(f'Redis URL: {redis_url}')

try:
    import redis
    r = redis.from_url(redis_url)
    r.ping()
    print('✅ Connexion à Redis réussie')
except Exception as e:
    print(f'⚠️  Erreur de connexion à Redis: {e}')
"
}

# Fonction de vérification des dépendances critiques
check_dependencies() {
    echo "📦 Vérification des dépendances critiques..."
    
    python -c "
import logging
logging.basicConfig(level=logging.INFO)

dependencies = [
    'flask', 'sqlalchemy', 'celery', 'redis', 'pandas', 
    'numpy', 'scikit-learn', 'torch', 'gymnasium'
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f'✅ {dep}')
    except ImportError:
        print(f'❌ {dep} manquant')
"
}

# Fonction de démarrage de l'application
start_application() {
    echo "🌐 Démarrage de l'application Flask..."
    
    # Choisir le mode de démarrage selon l'environnement
    if [ "$FLASK_ENV" = "development" ]; then
        echo "  Mode développement: démarrage avec Flask dev server"
        exec python app.py
    else
        echo "  Mode production: démarrage avec Gunicorn"
        exec gunicorn wsgi:app \
            --bind 0.0.0.0:5000 \
            --worker-class eventlet \
            --workers 1 \
            --timeout 120 \
            --keep-alive 2 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --preload \
            --access-logfile - \
            --error-logfile - \
            --log-level info
    fi
}

# Fonction de nettoyage à l'arrêt
cleanup() {
    echo "🧹 Nettoyage avant arrêt..."
    # Nettoyage des fichiers temporaires
    rm -rf /tmp/* /var/tmp/*
    echo "✅ Nettoyage terminé"
}

# Gestionnaire de signaux pour un arrêt propre
trap cleanup SIGTERM SIGINT

# Exécution des vérifications et du warmup
echo "🔍 Vérifications préliminaires..."

check_dependencies
check_database
check_redis
warmup_models

echo "✅ Toutes les vérifications terminées"
echo "🚀 Démarrage de l'application..."

# Démarrage de l'application
start_application
