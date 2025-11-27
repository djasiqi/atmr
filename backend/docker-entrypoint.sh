#!/usr/bin/env bash
# docker-entrypoint.sh
# Script d'entrée Docker avec warmup des modèles ML et vérifications de santé

# ⚠️ IMPORTANT: Ne pas utiliser 'set -e' ici car on veut gérer les erreurs de permissions gracieusement
set -uo pipefail

# Configuration des logs (créer le répertoire si nécessaire)
mkdir -p /app/logs 2>/dev/null || true
# Rediriger les logs vers un fichier si possible, sinon vers stdout/stderr
if [ -w /app/logs ] 2>/dev/null; then
    exec > >(tee -a /app/logs/docker-entrypoint.log 2>/dev/null || cat)
    exec 2>&1
else
    # Si on ne peut pas écrire dans /app/logs, utiliser stdout/stderr uniquement
    echo "⚠️  Impossible d'écrire dans /app/logs, utilisation de stdout/stderr uniquement"
fi

echo "🚀 Démarrage du conteneur ATMR Backend..."
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
# free n'est pas disponible dans l'image slim, utiliser une alternative
if command -v free >/dev/null 2>&1; then
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
else
    echo "Memory: N/A (free command not available)"
fi
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

# ⚠️ CRITIQUE: Empêcher pytest de s'exécuter automatiquement en production
# pytest peut être déclenché automatiquement lors de l'import de modules si pytest.ini existe
# ou si pytest est dans le PYTHONPATH. On désactive tout cela en production.
if [ "$FLASK_ENV" != "development" ] && [ "$FLASK_ENV" != "testing" ]; then
    echo "🔒 Mode production détecté - Désactivation de pytest..."
    
    # 1. Désactiver pytest.ini
    if [ -f /app/pytest.ini ]; then
        echo "  ⚠️  Désactivation de pytest.ini"
        mv /app/pytest.ini /app/pytest.ini.disabled 2>/dev/null || true
    fi
    
    # 2. Renommer le répertoire tests pour empêcher pytest de trouver les tests
    if [ -d /app/tests ] && [ ! -d /app/tests.disabled ]; then
        echo "  ⚠️  Désactivation du répertoire tests"
        mv /app/tests /app/tests.disabled 2>/dev/null || true
    fi
    
    # 3. Définir une variable d'environnement pour empêcher pytest de s'exécuter
    export PYTEST_DISABLED=1
    export DISABLE_PYTEST=1
    
    # 4. S'assurer que pytest n'est pas dans le PATH (si possible)
    # Note: On ne peut pas modifier le PATH système, mais on peut vérifier
    echo "  ✅ pytest désactivé pour la production"
else
    echo "  ℹ️  Mode $FLASK_ENV - pytest peut être utilisé"
fi

# Créer les répertoires de données nécessaires AVANT le warmup
# ⚠️ IMPORTANT: Les volumes Docker peuvent avoir des permissions root
# On essaie de créer les répertoires, mais on continue même en cas d'échec
# Les répertoires devraient être créés par le script de déploiement sur l'hôte
echo "📁 Création des répertoires de données..."
mkdir -p /app/data/ml /app/data/ml/models /app/data/rl /app/data/rl/shadow_mode 2>/dev/null || {
    echo "⚠️  Impossible de créer /app/data/* (permissions insuffisantes ou volume monté)"
    echo "   Les répertoires devraient être créés par le script de déploiement"
}
mkdir -p /app/logs /app/cache 2>/dev/null || {
    echo "⚠️  Impossible de créer /app/logs ou /app/cache (permissions insuffisantes)"
}
mkdir -p /app/uploads/company_logos 2>/dev/null || {
    echo "⚠️  Impossible de créer /app/uploads/company_logos (permissions insuffisantes)"
}

# S'assurer que les répertoires ont les bonnes permissions (si on a les droits)
# Utiliser 777 temporairement pour éviter les problèmes de permissions avec les volumes Docker
# En production, vous devriez utiliser un utilisateur non-root et des permissions plus restrictives
chmod -R 777 /app/data /app/logs /app/cache /app/uploads 2>/dev/null || {
    echo "⚠️  Impossible de modifier les permissions (normal si volumes montés avec root)"
}

# S'assurer que le répertoire models existe et a les bonnes permissions
if [ -d /app/data/ml/models ]; then
    chmod -R 755 /app/data/ml/models 2>/dev/null || true
fi

# S'assurer que le répertoire uploads/company_logos existe et a les bonnes permissions
if [ -d /app/uploads/company_logos ]; then
    chmod -R 755 /app/uploads/company_logos 2>/dev/null || true
fi

# Vérifier que les répertoires critiques existent (créés par le script de déploiement ou volumes)
if [ ! -d /app/data ]; then
    echo "❌ ERREUR CRITIQUE: /app/data n'existe pas"
    echo "   Le volume Docker backend_data doit être monté et le répertoire créé sur l'hôte"
    echo "   Vérifiez que le script de déploiement crée les répertoires nécessaires"
    exit 1
fi

if [ ! -d /app/logs ]; then
    echo "❌ ERREUR CRITIQUE: /app/logs n'existe pas"
    echo "   Le volume Docker backend_logs doit être monté"
    exit 1
fi

if [ ! -d /app/cache ]; then
    echo "❌ ERREUR CRITIQUE: /app/cache n'existe pas"
    echo "   Le volume Docker backend_cache doit être monté"
    exit 1
fi

echo "✅ Répertoires de données vérifiés"

# Essayer de changer le propriétaire si possible (peut échouer selon la configuration Docker)
# L'utilisateur par défaut dans l'image est généralement root ou un UID spécifique
if [ -n "${APP_USER:-}" ]; then
    chown -R "${APP_USER}" /app/data /app/logs /app/cache 2>/dev/null || true
fi

echo "✅ Répertoires de données créés avec permissions"

# Fonction de warmup des modèles ML
warmup_models() {
    echo "🔥 Warmup des modèles ML..."
    
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
        WORKERS="${GUNICORN_WORKERS:-4}"
        echo "  Workers configurés: $WORKERS"
        exec gunicorn wsgi:app \
            --bind 0.0.0.0:5000 \
            --worker-class eventlet \
            --workers "$WORKERS" \
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
