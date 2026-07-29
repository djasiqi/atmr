#!/usr/bin/env bash
# docker-entrypoint.sh
# Script d'entrée Docker avec warmup des modèles ML et vérifications de santé
#
# 🔍 STRATÉGIE DE GESTION DES ERREURS :
#
# ✅ ERREURS CRITIQUES (exit 1 - démarrage échoue) :
#    - Dépendances Python critiques manquantes (flask, sqlalchemy, etc.)
#    - DATABASE_URL manquante en production
#    - Base de données inaccessible en production
#    - Dépendances RL manquantes si RL_ENABLED=true
#
# ⚠️ ERREURS NON-CRITIQUES (exit 0 - démarrage continue) :
#    - Redis inaccessible (fallback vers memory storage)
#    - Modèles ML non chargés (optionnels)
#    - Scalers ML manquants (optionnels)
#    - DATABASE_URL manquante en développement/testing
#
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

# Sous-dossiers uploads métier (PDF factures, logos, etc.) — doivent rester
# inscriptibles par appuser. Ne jamais appliquer chmod 755 récursif sur
# /app/uploads : si le chown échoue (volume bind Windows/NFS), 755 bloque
# l'écriture → [Errno 13] Permission denied à la génération PDF.
UPLOAD_SUBDIRS="company_logos invoices statements institution_logos chat"
APP_UID="$(id -u appuser 2>/dev/null || echo 999)"
APP_GID="$(id -g appuser 2>/dev/null || echo 999)"

# Si on est root, corriger les permissions avant de passer à appuser
if [ "$(id -u)" = "0" ]; then
    echo "🔐 Correction des permissions en tant que root (appuser=${APP_UID}:${APP_GID})..."
    mkdir -p /app/data /app/data/ml /app/data/ml/models /app/data/rl /app/data/rl/shadow_mode
    mkdir -p /app/logs /app/cache
    for sub in $UPLOAD_SUBDIRS; do
        mkdir -p "/app/uploads/$sub"
    done
    chmod -R 755 /app/data /app/logs /app/cache 2>/dev/null || true
    chown -R "${APP_UID}:${APP_GID}" /app/data /app/logs /app/cache /app/uploads 2>/dev/null || true
    # uploads : owner/group RWX — pas 755 qui retire le write groupe
    chmod -R u+rwX,g+rwX,o+rX /app/uploads 2>/dev/null || true
    # Bind mounts Windows/NFS : si toujours non inscriptible après chown, ouvrir en a+rwX
    for sub in $UPLOAD_SUBDIRS; do
        if [ -d "/app/uploads/$sub" ] && [ ! -w "/app/uploads/$sub" ]; then
            echo "⚠️  /app/uploads/$sub non inscriptible après chown — chmod a+rwX"
            chmod -R a+rwX "/app/uploads/$sub" 2>/dev/null || true
        fi
    done
    # Corriger les permissions de tous les fichiers .pkl existants
    find /app/data/ml/models -name "*.pkl" -type f -exec chmod 644 {} \; -exec chown "${APP_UID}:${APP_GID}" {} \; 2>/dev/null || true
    echo "✅ Permissions corrigées"
    echo "📋 Vérification des permissions de /app/data/ml/models:"
    ls -la /app/data/ml/models/ 2>/dev/null || echo "  Répertoire non accessible"
    echo "📋 Vérification des permissions de /app/uploads:"
    ls -la /app/uploads/ 2>/dev/null || echo "  Répertoire non accessible"
    # Passer à appuser si on est root
    exec gosu appuser "$0" "$@"
fi

# Si on est appuser, essayer de créer les répertoires (peut échouer silencieusement)
mkdir -p /app/data/ml /app/data/ml/models /app/data/rl /app/data/rl/shadow_mode 2>/dev/null || {
    echo "⚠️  Impossible de créer /app/data/* (permissions insuffisantes ou volume monté)"
    echo "   Les répertoires devraient être créés par le script de déploiement"
}
mkdir -p /app/logs /app/cache 2>/dev/null || {
    echo "⚠️  Impossible de créer /app/logs ou /app/cache (permissions insuffisantes)"
}
for sub in $UPLOAD_SUBDIRS; do
    mkdir -p "/app/uploads/$sub" 2>/dev/null || {
        echo "⚠️  Impossible de créer /app/uploads/$sub (permissions insuffisantes)"
    }
done

# data/logs/cache : 755 OK (lecture/exé pour tous, écriture owner)
chmod -R 755 /app/data /app/logs /app/cache 2>/dev/null || {
    echo "⚠️  Impossible de modifier les permissions data/logs/cache (normal si volumes montés avec root)"
}

# S'assurer que le répertoire models existe et a les bonnes permissions
if [ -d /app/data/ml/models ]; then
    chmod -R 755 /app/data/ml/models 2>/dev/null || true
fi

# uploads : conserver l'écriture pour le propriétaire (ne pas forcer 755)
if [ -d /app/uploads ]; then
    chmod -R u+rwX,g+rwX /app/uploads 2>/dev/null || {
        echo "⚠️  Impossible d'assurer l'écriture sur /app/uploads (volume monté ?)"
    }
fi

# Alerte précoce si la génération PDF échouera (Errno 13)
if [ -d /app/uploads/invoices ] && [ ! -w /app/uploads/invoices ]; then
    echo "❌ /app/uploads/invoices non inscriptible par $(whoami) — génération PDF factures en échec"
    echo "   Corriger: chown -R ${APP_UID}:${APP_GID} /app/uploads && chmod -R a+rwX /app/uploads"
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
    # Vérifier plusieurs emplacements possibles
    MODEL_PATH=""
    if [ -f "/app/data/ml/models/delay_predictor.pkl" ]; then
        MODEL_PATH="/app/data/ml/models/delay_predictor.pkl"
    elif [ -f "/app/data/ml/delay_predictor.pkl" ]; then
        MODEL_PATH="/app/data/ml/delay_predictor.pkl"
    fi
    
    if [ -n "$MODEL_PATH" ]; then
        echo "  📊 Chargement du modèle de prédiction de retard depuis $MODEL_PATH..."
        python -c "
import pickle
import logging
logging.basicConfig(level=logging.INFO)
try:
    with open('$MODEL_PATH', 'rb') as f:
        model = pickle.load(f)
    print(f'✅ Modèle de prédiction de retard chargé: {type(model).__name__}')
except Exception as e:
    print(f'⚠️  Erreur lors du chargement du modèle de prédiction: {e}')

import sys
sys.exit(0)
"
    else
        echo "  ℹ️  Modèle de prédiction de retard non trouvé (optionnel)"
    fi
    
    # Warmup des modèles RL (uniquement si RL_ENABLED=true)
    rl_enabled=$(python -c "import os; print('true' if os.getenv('RL_ENABLED', 'false').lower() in ('true', '1', 'yes') else 'false')")
    if [ "$rl_enabled" = "true" ] && [ -f "/app/data/rl/best_model.pth" ]; then
        echo "  🤖 Chargement du modèle RL (RL_ENABLED=true)..."
        python -c "
import os
import logging
logging.basicConfig(level=logging.INFO)

# Vérifier que RL est activé
if not os.getenv('RL_ENABLED', 'false').lower() in ('true', '1', 'yes'):
    print('ℹ️  RL désactivé – skip warmup RL')
    exit(0)

try:
    import torch
    model = torch.load('/app/data/rl/best_model.pth', map_location='cpu')
    print(f'✅ Modèle RL chargé: {type(model).__name__}')
    # Test d'inférence pour vérifier le modèle
    if hasattr(model, 'forward'):
        dummy_input = torch.randn(1, 10)  # Exemple d'input
        with torch.no_grad():
            _ = model(dummy_input)
        print('✅ Test d\'inférence RL réussi')
except ImportError as e:
    print(f'⚠️  PyTorch non disponible (RL_ENABLED=true mais torch manquant): {e}')
except Exception as e:
    print(f'⚠️  Erreur lors du chargement du modèle RL: {e}')

import sys
sys.exit(0)
"
    elif [ "$rl_enabled" = "false" ]; then
        echo "  ℹ️  RL désactivé (RL_ENABLED=false) – skip warmup RL"
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

import sys
sys.exit(0)
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
import sys
import logging
logging.basicConfig(level=logging.INFO)

flask_env = os.getenv('FLASK_ENV', 'development')
is_production = flask_env == 'production'

# Vérifier les variables d'environnement de la DB
db_url = os.getenv('DATABASE_URL', '')
if db_url:
    print(f'✅ DATABASE_URL configurée: {db_url[:20]}...')
else:
    print('⚠️  DATABASE_URL non configurée')
    if is_production:
        print('❌ ERREUR CRITIQUE: DATABASE_URL obligatoire en production')
        sys.exit(1)  # Échec en production si DB manquante

# Test de connexion si possible
try:
    from sqlalchemy import create_engine, text
    if db_url:
        engine = create_engine(db_url, pool_pre_ping=True, connect_args={'connect_timeout': 5})
        with engine.connect() as conn:
            result = conn.execute(text('SELECT 1'))
            print('✅ Connexion à la base de données réussie')
except Exception as e:
    print(f'⚠️  Erreur de connexion à la base de données: {e}')
    if is_production:
        print('❌ ERREUR CRITIQUE: Impossible de se connecter à la DB en production')
        sys.exit(1)  # Échec en production si DB inaccessible

sys.exit(0)  # Succès
"
}

# Fonction de vérification de Redis
check_redis() {
    echo "🔴 Vérification de Redis..."
    
    python -c "
import os
import sys
import time
import logging
logging.basicConfig(level=logging.INFO)

# Utiliser REDIS_URL en priorité (configuré dans le workflow de déploiement)
# Sinon CELERY_BROKER_URL, sinon construire depuis REDIS_HOST si disponible
redis_url = os.getenv('REDIS_URL') or os.getenv('CELERY_BROKER_URL')
if not redis_url:
    # Construire depuis REDIS_HOST si disponible, sinon fallback
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = os.getenv('REDIS_PORT', '6379')
    redis_db = os.getenv('REDIS_DB', '0')
    redis_password = os.getenv('REDIS_PASSWORD', '')
    
    if redis_password:
        from urllib.parse import quote_plus
        redis_password_escaped = quote_plus(redis_password)
        redis_url = f'redis://:{redis_password_escaped}@{redis_host}:{redis_port}/{redis_db}'
    else:
        redis_url = f'redis://{redis_host}:{redis_port}/{redis_db}'

print(f'Redis URL: {redis_url}')

max_attempts = int(os.getenv('REDIS_STARTUP_MAX_ATTEMPTS', '90'))
sleep_sec = float(os.getenv('REDIS_STARTUP_RETRY_SLEEP_SEC', '2'))

try:
    import redis
    import redis.exceptions as redis_exc
except ImportError:
    print('⚠️  Package redis non importable')
    sys.exit(0)

last_err = None
for attempt in range(1, max_attempts + 1):
    try:
        r = redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5)
        r.ping()
        print('✅ Connexion à Redis réussie')
        sys.exit(0)
    except redis_exc.BusyLoadingError as e:
        last_err = e
    except redis_exc.ConnectionError as e:
        last_err = e
    except redis_exc.TimeoutError as e:
        last_err = e
    except redis_exc.ResponseError as e:
        last_err = e
        err_s = str(e).upper()
        if 'LOADING' not in err_s and 'LOADING Redis' not in err_s:
            break
    except OSError as e:
        last_err = e
    except Exception as e:
        last_err = e
        break

    if attempt < max_attempts:
        print(f'⏳ Redis indisponible ou en chargement (tentative {attempt}/{max_attempts}): {last_err}')
        time.sleep(sleep_sec)

# ⚠️  Redis est NON-CRITIQUE: le backend peut démarrer avec fallback memory
print(f'⚠️  Redis non joignable après {max_attempts} tentatives (non-critique, fallback memory): {last_err}')
sys.exit(0)
"
}

# Fonction de vérification des dépendances critiques
check_dependencies() {
    echo "📦 Vérification des dépendances critiques..."
    
    python -c "
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

try:
    # Vérifier si RL est activé
    rl_enabled = os.getenv('RL_ENABLED', 'false').lower() in ('true', '1', 'yes')
    with_rl = os.getenv('WITH_RL', 'false').lower() in ('true', '1', 'yes')
    rl_active = rl_enabled or with_rl

    if not rl_active:
        print('ℹ️  RL désactivé dans cet environnement – on ignore les dépendances RL.')
        print('   (torch, gymnasium, optuna ne sont pas installés en production)')

    # Dépendances critiques (toujours requises)
    critical_deps = ['flask', 'sqlalchemy', 'celery', 'redis', 'pandas', 'numpy', 'sklearn']
    missing_critical = []

    # Vérifier les dépendances critiques
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f'✅ {dep}')
        except ImportError as e:
            print(f'❌ {dep} manquant: {e}')
            missing_critical.append(dep)
        except Exception as e:
            print(f'⚠️  Erreur lors de l\'import de {dep}: {e}')
            missing_critical.append(dep)

    # Faire échouer si des dépendances critiques manquent
    if missing_critical:
        print(f'❌ ERREUR CRITIQUE: {len(missing_critical)} dépendance(s) critique(s) manquante(s): {missing_critical}')
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)

    # Vérifier les dépendances ML uniquement si RL est activé
    if rl_active:
        print('📦 Vérification des dépendances RL/ML...')
        ml_deps = ['torch', 'gymnasium', 'optuna']
        missing_ml = []
        for dep in ml_deps:
            try:
                __import__(dep)
                print(f'✅ {dep}')
            except ImportError as e:
                print(f'❌ {dep} manquant (requis pour RL): {e}')
                missing_ml.append(dep)
            except Exception as e:
                print(f'⚠️  Erreur lors de l\'import de {dep}: {e}')
                missing_ml.append(dep)
        
        # Faire échouer si RL activé mais dépendances manquantes
        if missing_ml:
            print(f'❌ ERREUR CRITIQUE: RL activé mais {len(missing_ml)} dépendance(s) RL manquante(s): {missing_ml}')
            sys.exit(1)
    else:
        print('ℹ️  Dépendances RL ignorées (RL_ENABLED=false)')

    print('✅ Toutes les dépendances critiques sont présentes')
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # Succès

except Exception as e:
    print(f'❌ ERREUR FATALE lors de la vérification des dépendances: {e}')
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(1)
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
        # ✅ FIX Socket.IO multi-workers: Pour diagnostiquer "Invalid session" errors,
        # définir GUNICORN_WORKERS=1 pour forcer un seul worker (évite le problème de SID
        # partagé entre workers).
        # ⚠️ gevent + Flask-SocketIO : NE PAS utiliser --preload.
        # Avec --preload, wsgi:app est importé dans le master AVANT fork, donc
        # gevent.monkey.patch_all() (voir wsgi.py) s'exécute avant le fork et laisse
        # le hub gevent / locks dans un état incohérent dans les workers. Conséquence :
        # freeze non coopératif au teardown du recyclage --max-requests (graceful-timeout
        # ignoré, WORKER TIMEOUT après --timeout). Sans --preload, chaque worker importe
        # wsgi:app après fork et patche proprement (cf. wsgi.py:61-70).
        # Défaut 2 workers (REST + Socket.IO sur le même pool) ; augmenter via GUNICORN_WORKERS si besoin.
        WORKERS="${GUNICORN_WORKERS:-2}"
        GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-180}"
        GUNICORN_GRACEFUL_TIMEOUT="${GUNICORN_GRACEFUL_TIMEOUT:-30}"
        GUNICORN_KEEPALIVE="${GUNICORN_KEEPALIVE:-5}"
        echo "  Workers configurés: $WORKERS"
        echo "  Timeout worker: ${GUNICORN_TIMEOUT}s"
        echo "  Graceful timeout: ${GUNICORN_GRACEFUL_TIMEOUT}s"
        if [ "$WORKERS" = "1" ]; then
            echo "  ⚠️  Mode single-worker (diagnostic Socket.IO multi-workers)"
        fi
        # Gunicorn 26+ ne fournit plus le worker eventlet ; défaut gevent.
        _skip_sio="false"
        case "$(echo "${SKIP_SOCKETIO:-false}" | tr '[:upper:]' '[:lower:]')" in
            1|true|yes) _skip_sio="true" ;;
        esac
        _gw="${GUNICORN_WORKER_CLASS:-gevent}"
        if [ "$_skip_sio" = "true" ]; then
            _gw="${GUNICORN_WORKER_CLASS:-gthread}"
            echo "  Mode REST pur (SKIP_SOCKETIO=1) — worker défaut: $_gw"
        fi
        if [ "$_gw" = "eventlet" ]; then
            echo "⚠️  GUNICORN_WORKER_CLASS=eventlet incompatible avec Gunicorn 26+ — bascule sur gevent."
            _gw=gevent
        fi
        _gunicorn_extra=()
        if [ "$_gw" = "gthread" ]; then
            _threads="${GUNICORN_THREADS:-4}"
            _gunicorn_extra=(--threads "$_threads")
            echo "  Threads gthread: $_threads"
        fi
        exec gunicorn wsgi:app \
            --config gunicorn.conf.py \
            --bind 0.0.0.0:5000 \
            --worker-class "$_gw" \
            --workers "$WORKERS" \
            "${_gunicorn_extra[@]}" \
            --timeout "$GUNICORN_TIMEOUT" \
            --graceful-timeout "$GUNICORN_GRACEFUL_TIMEOUT" \
            --keep-alive "$GUNICORN_KEEPALIVE" \
            --worker-tmp-dir /dev/shm \
            --max-requests 1000 \
            --max-requests-jitter 250 \
            --access-logfile - \
            --error-logfile - \
            --capture-output \
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

# Migrations Alembic dans le conteneur
# En production avec PgBouncer : appliquer les migrations hors entrypoint avec une URI
# directe Postgres (voir deploy-production.sh migration_exec — DATABASE_URL_DIRECT).
# Sinon Alembic peut échouer (session / pool transactionnel / locks) alors que deploy
# relance ensuite un upgrade avec la bonne cible → double passe et logs trompeurs.
run_migrations() {
    echo "🔄 Migrations Alembic (entrypoint)..."

    if [ -z "${DATABASE_URL:-}" ]; then
        echo "ℹ️  DATABASE_URL non configurée, migrations ignorées"
        return 0
    fi

    if [ "${RUN_ENTRYPOINT_MIGRATIONS:-}" = "0" ] ||
        [ "${RUN_ENTRYPOINT_MIGRATIONS:-}" = "false" ] ||
        [ "${RUN_ENTRYPOINT_MIGRATIONS:-}" = "no" ]; then
        echo "ℹ️  RUN_ENTRYPOINT_MIGRATIONS désactivé — migrations déléguées au déploiement (prod recommandé)"
        return 0
    fi

    # Détection défensive même si DATABASE_URL était surchargée manuellement
    if [[ -z "${MIGRATE_DATABASE_URL:-}" ]]; then
        case "${DATABASE_URL}" in
            *@pgbouncer:* | *://*@pgbouncer/* | *@pgbouncer/*)
                echo "ℹ️  PgBouncer détecté dans DATABASE_URL — pas d'upgrade Alembic ici (utiliser Postgres direct ou MIGRATE_DATABASE_URL)"
                return 0
                ;;
            *:6432/*)
                echo "ℹ️  Port 6432 (souvent PgBouncer) dans DATABASE_URL — migrations entrypoint ignorées"
                return 0
                ;;
        esac
    fi

    local saved_db_url saved_sqlalchemy saved_primary
    saved_db_url="${DATABASE_URL:-}"
    saved_sqlalchemy="${SQLALCHEMY_DATABASE_URI:-}"
    saved_primary="${PRIMARY_DATABASE_URL:-}"

    if [ -n "${MIGRATE_DATABASE_URL:-}" ]; then
        echo "  Utilisation de MIGRATE_DATABASE_URL pour Alembic (connexion directe dédiée)"
        export DATABASE_URL="$MIGRATE_DATABASE_URL"
        export SQLALCHEMY_DATABASE_URI="$MIGRATE_DATABASE_URL"
        export PRIMARY_DATABASE_URL="$MIGRATE_DATABASE_URL"
    fi

    # eventlet doit rester désactivé pour les transactions Alembic (voir manage.py)
    export DISABLE_EVENTLET=1
    if flask db upgrade heads; then
        echo "✅ Migrations appliquées (entrypoint)"
    else
        echo "❌ Échec: flask db upgrade heads (entrypoint)"
        if [ "$FLASK_ENV" = "production" ]; then
            exit 1
        fi
        echo "⚠️  Mode non-production: poursuite du démarrage malgré l'échec des migrations"
    fi

    if [ -n "${MIGRATE_DATABASE_URL:-}" ]; then
        export DATABASE_URL="$saved_db_url"
        [ -n "$saved_sqlalchemy" ] && export SQLALCHEMY_DATABASE_URI="$saved_sqlalchemy" || unset SQLALCHEMY_DATABASE_URI
        [ -n "$saved_primary" ] && export PRIMARY_DATABASE_URL="$saved_primary" || unset PRIMARY_DATABASE_URL
    fi

    return 0
}
run_migrations

echo "✅ Toutes les vérifications terminées"
echo "🚀 Démarrage de l'application..."

# Démarrage de l'application
start_application
