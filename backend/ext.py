# backend/ext.py
# pyright: reportImportCycles = false

import logging
import os
from functools import wraps
from typing import Any, Literal, cast

import redis
from flask import abort, jsonify, request
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

# Initialisation des extensions (singleton importables partout)
db = SQLAlchemy()
jwt = JWTManager()
mail = Mail()
bcrypt = Bcrypt()
migrate = Migrate()

# Redis
REDIS_URL = os.getenv("REDIS_URL") or "redis://redis:6379/0"

try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.ping()
    limiter_storage = REDIS_URL
except Exception:
    redis_client = None
    limiter_storage = "memory://"

# ⚠️ Ne fixe PAS CORS/path ici pour éviter les conflits - tout est défini dans app.py (socketio.init_app)
# Active la file Redis si disponible (scaling multi-workers).
# Typage strict de async_mode pour contenter Pylance/pyright.
AsyncMode = Literal["threading", "eventlet", "gevent", "gevent_uwsgi"]
_env_async = (os.getenv("SOCKETIO_ASYNC_MODE") or "eventlet").strip().lower()
_allowed_modes = {"threading", "eventlet", "gevent", "gevent_uwsgi"}
if _env_async not in _allowed_modes:
    # fallback sûr si une valeur inconnue est fournie
    _env_async = "eventlet"
ASYNC_MODE: AsyncMode = cast("AsyncMode", _env_async)

socketio = SocketIO(
    async_mode=ASYNC_MODE,
    message_queue=REDIS_URL if redis_client else None,
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["5000 per hour"],
    storage_uri=limiter_storage,
)

dispatch_status = {"is_running": False, "last_run_time": None}
app_logger = logging.getLogger("app")

# ✅ 2.7: Constantes pour DB Profiler
DB_PROFILER_N_PLUS_1_THRESHOLD = 10  # Seuil pour détecter pattern N+1
DB_PROFILER_WARNING_QUERY_COUNT = 15  # Seuil pour avertir trop de requêtes


# Event listener pour compter les requêtes SQL (performance metrics)
def setup_sql_event_listener():
    """Configure l'event listener SQLAlchemy pour compter les requêtes SQL."""
    from sqlalchemy import engine
    from sqlalchemy import event as sqlalchemy_event

    from services.unified_dispatch import performance_metrics

    def _receive_before_cursor_execute(_conn, _cursor, statement, _parameters, _context, _executemany):
        """Compteur SQL pour performance metrics."""
        # Détecter type de requête
        query_type = "SELECT"  # par défaut
        stmt_upper = statement.upper().strip()
        if stmt_upper.startswith("INSERT"):
            query_type = "INSERT"
        elif stmt_upper.startswith("UPDATE"):
            query_type = "UPDATE"
        elif stmt_upper.startswith("DELETE"):
            query_type = "DELETE"
        elif stmt_upper.startswith("SELECT"):
            query_type = "SELECT"

        # Incrémenter compteur
        performance_metrics.increment_sql_counter(query_type)

    sqlalchemy_event.listens_for(engine.Engine, "before_cursor_execute")(_receive_before_cursor_execute)


# ✅ 2.7: Setup DB Profiler pour détecter N+1
def setup_db_profiler(app):
    """Configure le profiler DB si activé via ENABLE_DB_PROFILING.

    Args:
        app: Instance Flask
    """
    from shared.db_profiler import get_db_profiler, is_profiling_enabled

    if is_profiling_enabled():
        profiler = get_db_profiler()
        app.logger.info("[DB Profiler] ✅ Profiling DB activé via ENABLE_DB_PROFILING")

        # Middleware pour profiler chaque requête HTTP
        @app.before_request
        def _start_profiling():  # pyright: ignore[reportUnusedFunction]
            """Démarre le profiling pour cette requête."""
            if profiler.enabled:
                profiler.reset()

        @app.after_request
        def _end_profiling(response):  # pyright: ignore[reportUnusedFunction]
            """Termine le profiling et log les statistiques."""
            if profiler.enabled:
                stats = profiler.get_stats()

                # Log si trop de requêtes (suspect N+1)
                if stats["query_count"] > DB_PROFILER_WARNING_QUERY_COUNT:
                    app.logger.warning(
                        "[DB Profiler] ⚠️ %d requêtes sur %s %s (suspect N+1?)",
                        stats["query_count"],
                        request.method,
                        request.path,
                    )

                # Détecter N+1
                if profiler.detect_n_plus_1(threshold=DB_PROFILER_N_PLUS_1_THRESHOLD):
                    app.logger.error("[DB Profiler] 🚨 PATTERN N+1 détecté sur %s %s", request.method, request.path)

                # Ajouter headers de profiling si demandé
                if os.getenv("DB_PROFILING_HEADERS", "false").lower() == "true":
                    response.headers["X-DB-Query-Count"] = str(stats["query_count"])
                    response.headers["X-DB-Total-Time-Ms"] = str(stats["total_time_ms"])

            return response

        app.logger.info("[DB Profiler] Middleware de profiling configuré")
    else:
        app.logger.debug("[DB Profiler] Profiling désactivé (set ENABLE_DB_PROFILING=true)")


#  JWT error handlers
@jwt.expired_token_loader
def expired_token_callback(_jwt_header, _jwt_payload):
    return jsonify({"error": "Le token a expiré. Veuillez vous reconnecter."}), 401


@jwt.invalid_token_loader
def invalid_token_callback(_error):
    return jsonify({"error": "Token invalide. Veuillez vous reconnecter."}), 422


@jwt.unauthorized_loader
def missing_token_callback(_error):
    return jsonify({"error": "Token d'accès manquant. Veuillez vous authentifier."}), 401


@jwt.needs_fresh_token_loader
def token_not_fresh_callback(_jwt_header, _jwt_payload):
    return jsonify({"error": "Le token n'est pas frais. Veuillez vous reconnecter."}), 401


# ✅ Phase 3: Callback pour vérifier la blacklist des tokens
@jwt.token_in_blocklist_loader
def check_if_token_revoked(_jwt_header, jwt_payload):
    """Vérifie si le token JWT est dans la blacklist.

    Args:
        _jwt_header: En-tête JWT (non utilisé)
        jwt_payload: Payload JWT

    Returns:
        True si le token est révoqué (blacklisté), False sinon
    """
    from security.token_blacklist import is_token_blacklisted

    jti = jwt_payload.get("jti")
    if jti:
        return is_token_blacklisted(jti=jti)

    return False


# ✅ Hardening JWT: Validation explicite de l'audience (défense en profondeur)
@jwt.additional_claims_loader
def add_claims_to_access_token(_identity):
    """Ajoute des claims supplémentaires au token (déjà fait dans auth.py, mais défense en profondeur).

    Args:
        _identity: Identité de l'utilisateur (non utilisé, mais requis par Flask-JWT-Extended)
    """
    # Les claims sont déjà ajoutés dans auth.py lors de la création du token
    # Ce callback est optionnel mais peut servir de vérification supplémentaire
    return {}


def validate_jwt_audience(jwt_payload: dict[str, Any]) -> bool:
    """Valide que l'audience du token JWT correspond à l'audience attendue.

    Args:
        jwt_payload: Payload JWT décodé

    Returns:
        True si l'audience est valide, False sinon
    """
    expected_audience = "atmr-api"
    token_audience = jwt_payload.get("aud")

    if not token_audience:
        app_logger.warning("[JWT Security] Token sans claim 'aud' (audience)")
        return False

    if token_audience != expected_audience:
        app_logger.warning("[JWT Security] Audience invalide: %s (attendu: %s)", token_audience, expected_audience)
        return False

    return True


#  Decorator role_required
def role_required(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            from models import User  # import local pour éviter les cycles

            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()

            if not user:
                app_logger.warning("Utilisateur non trouvé pour le public_id : %s", user_public_id)
                abort(404, description="Utilisateur non trouvé")

            # Convertir les rôles en objets UserRole pour la comparaison
            from models import UserRole

            allowed_roles = []

            # Gérer les deux formats : @role_required(['ADMIN', 'COMPANY']) et @role_required(UserRole.company)
            if roles and len(roles) > 0:
                first_arg = roles[0]
                if isinstance(first_arg, list):
                    # Format: @role_required(['ADMIN', 'COMPANY'])
                    role_list = first_arg
                elif hasattr(first_arg, "value"):
                    # Format: @role_required(UserRole.company)
                    role_list = [first_arg.value]
                else:
                    # Format: @role_required('COMPANY')
                    role_list = [first_arg]

                for role_str in role_list:
                    try:
                        allowed_roles.append(UserRole[role_str])
                    except KeyError:
                        app_logger.warning("Rôle invalide dans la configuration : %s", role_str)

            if user.role not in allowed_roles:
                app_logger.warning(
                    "⛔ Accès refusé : %s (%s) a tenté d'accéder à une route restreinte.", user.username, user.role
                )
                abort(403, description="Accès non autorisé")

            return fn(*args, **kwargs)

        return wrapper

    return decorator
