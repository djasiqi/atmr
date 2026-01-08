# backend/ext.py
# pyright: reportImportCycles = false

import json
import logging
import os
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Literal, cast

import redis  # pyright: ignore[reportMissingImports]
from flask import (
    abort,
    jsonify,
    request,
)
from flask_bcrypt import Bcrypt  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    JWTManager,
    get_jwt_identity,
)
from flask_limiter import Limiter  # pyright: ignore[reportMissingImports]
from flask_limiter.util import (  # pyright: ignore[reportMissingImports]
    get_remote_address,
)
from flask_mail import Mail  # pyright: ignore[reportMissingImports]
from flask_migrate import Migrate  # pyright: ignore[reportMissingModuleSource]
from flask_socketio import SocketIO  # pyright: ignore[reportMissingModuleSource]
from flask_sqlalchemy import SQLAlchemy  # pyright: ignore[reportMissingImports]

# Initialisation des extensions (singleton importables partout)
db = SQLAlchemy()
jwt = JWTManager()
mail = Mail()
bcrypt = Bcrypt()
migrate = Migrate()

# Logger (défini tôt pour être disponible partout)
app_logger = logging.getLogger("app")

# Redis
REDIS_URL = os.getenv("REDIS_URL", default="redis://redis:6379/0")

# ✅ P1: Timeouts Redis pour éviter blocages si Redis indisponible
REDIS_SOCKET_TIMEOUT = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))  # 5s par défaut
REDIS_SOCKET_CONNECT_TIMEOUT = int(
    os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")
)  # 5s par défaut

redis_client: redis.Redis | None = None
try:
    # ✅ P1: Configurer timeouts pour éviter blocages
    redis_client = redis.Redis.from_url(
        REDIS_URL,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
    )
    # ✅ P1: Ping avec timeout pour vérifier connexion sans bloquer
    if redis_client is not None:
        redis_client.ping()
    limiter_storage = REDIS_URL
except Exception as e:
    app_logger.warning(
        "[Redis] Connexion échouée ou timeout: %s. Fallback vers memory storage.",
        e,
    )
    redis_client = None
    limiter_storage = "memory://"

# ⚠️ Ne fixe PAS CORS/path ici pour éviter les conflits
# - tout est défini dans app.py (socketio.init_app)
# ✅ FIX Socket.IO multi-workers: Active la file Redis si disponible
# (scaling multi-workers).
# CRITIQUE: Utiliser REDIS_URL directement (pas redis_client) car Flask-SocketIO
# se connectera à Redis quand nécessaire. Si redis_client est None au démarrage
# (Redis pas encore disponible), message_queue=None causerait "Invalid session"
# en multi-workers car les SIDs ne sont pas partagés entre workers.
# Typage strict de async_mode pour contenter Pylance/pyright.
AsyncMode = Literal["threading", "eventlet", "gevent", "gevent_uwsgi"]
_env_async = os.getenv("SOCKETIO_ASYNC_MODE", default="eventlet").strip().lower()
_allowed_modes = {"threading", "eventlet", "gevent", "gevent_uwsgi"}
if _env_async not in _allowed_modes:
    # fallback sûr si une valeur inconnue est fournie
    _env_async = "eventlet"
ASYNC_MODE: AsyncMode = cast("AsyncMode", _env_async)

# ✅ FIX: Utiliser REDIS_URL directement si disponible (même si redis_client est None)
# Flask-SocketIO se connectera à Redis automatiquement quand nécessaire.
# Si REDIS_URL est vide ou "memory://", alors message_queue=None (mode single-worker).
_socketio_message_queue: str | None = None
if REDIS_URL and REDIS_URL.strip() and not REDIS_URL.startswith("memory://"):
    _socketio_message_queue = REDIS_URL
    app_logger.info(
        "[Socket.IO] ✅ Message queue Redis configurée: %s (multi-workers support)",
        REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL,
    )
else:
    msg = (
        "[Socket.IO] ⚠️  Message queue Redis non configurée - "
        "mode single-worker uniquement. Pour multi-workers, définissez REDIS_URL."
    )
    app_logger.warning(msg)

socketio = SocketIO(
    async_mode=ASYNC_MODE,
    message_queue=_socketio_message_queue,
)

# #region agent log
try:
    # Utiliser variable d'environnement ou chemin par défaut (Windows/Docker)
    debug_log_path = os.getenv(
        "DEBUG_LOG_PATH", r"c:\Users\jasiq\atmr\.cursor\debug.log"
    )
    log_path = Path(debug_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        log_data = {
            "id": f"log_{int(datetime.now(UTC).timestamp() * 1000)}_socketio_init",
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
            "location": "ext.py:socketio_init",
            "message": "SocketIO initialized with message_queue",
            "data": {
                "async_mode": ASYNC_MODE,
                "message_queue_configured": _socketio_message_queue is not None,
                "message_queue": _socketio_message_queue.split("@")[-1]
                if _socketio_message_queue and "@" in _socketio_message_queue
                else (_socketio_message_queue or "None"),
                "worker_pid": os.getpid(),
                "redis_url_available": bool(REDIS_URL and REDIS_URL.strip()),
            },
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A",
        }
        f.write(json.dumps(log_data) + "\n")
except Exception:
    pass
# #endregion


# ========================
# ✅ C2: Redis Storage avec TTL automatiques
# ========================


class RedisStorageWithTTL:
    """
    Wrapper pour le storage Redis qui ajoute automatiquement des TTL aux clés.

    Cela évite l'accumulation de clés obsolètes en mémoire Redis et garantit
    un nettoyage automatique sans intervention manuelle.
    """

    def __init__(self, storage_uri: str, ttl_seconds: int = 7200):
        """
        Initialise le storage Redis avec TTL automatique.

        Args:
            storage_uri: URI de connexion Redis (ex: "redis://localhost:6379/0")
            ttl_seconds: Durée de vie des clés en secondes (défaut: 2 heures)
        """
        super().__init__()  # Appel du constructeur parent implicite (object)

        # Import tardif pour éviter les cycles
        from flask_limiter.storage import (  # pyright: ignore[reportMissingImports]
            RedisStorage,
        )

        self.storage = RedisStorage(storage_uri)
        self.ttl_seconds = ttl_seconds
        app_logger.info(
            "[Rate Limit] Storage Redis avec TTL automatique: %d secondes (%d heures)",
            ttl_seconds,
            ttl_seconds // 3600,
        )

    def __getattr__(self, name):
        """Délègue tous les appels au storage Redis sous-jacent."""
        return getattr(self.storage, name)

    def incr(
        self,
        key: str,
        expiry: int,
        elastic_expiry: bool = False,
        amount: int = 1,
    ) -> int:
        """
        Incrémente le compteur et définit un TTL sur la clé.

        Args:
            key: Clé Redis à incrémenter
            expiry: Expiration en secondes (paramètre Flask-Limiter, peut être ignoré)
            elastic_expiry: Si True, prolonge le TTL à chaque hit
            amount: Valeur d'incrémentation

        Returns:
            Nouvelle valeur du compteur
        """
        # Incrémenter via le storage sous-jacent
        result = self.storage.incr(key, expiry, elastic_expiry, amount)

        # ✅ Définir un TTL automatique sur la clé
        # Si elastic_expiry=True, prolonge le TTL à chaque requête
        # (comportement "sliding window")
        # Sinon, utilise le TTL configuré une seule fois
        try:
            redis_conn = self.storage.storage
            if (
                redis_conn
                and hasattr(redis_conn, "expire")
                and (elastic_expiry or result == amount)
            ):
                # Première création ou mode élastique : définir/prolonger le TTL
                redis_conn.expire(key, self.ttl_seconds)
        except Exception as e:
            # Ne pas crasher si le TTL échoue, juste logger
            app_logger.warning(
                "[Rate Limit] Échec définition TTL pour clé %s: %s", key, e
            )

        return result


# ✅ C2: Fonction pour générer un hash de configuration (versioning des clés Redis)
def get_rate_limit_config_hash() -> str:
    """Génère un hash unique basé sur les configurations de rate limit.

    Ce hash est inclus dans les clés Redis pour invalider automatiquement
    les anciens rate limits lorsque la configuration change.

    Returns:
        Hash de configuration (8 caractères hexadécimaux)
    """
    import hashlib

    # Récupérer la version depuis la config ou l'environnement
    version = os.getenv("RATELIMIT_CONFIG_VERSION", "v1")
    environment = os.getenv("ENVIRONMENT", "development")
    default_limits = os.getenv("RATELIMIT_DEFAULT_LIMITS", "1000 per hour")

    config_str = f"{version}:{environment}:{default_limits}"
    # MD5 utilisé uniquement pour générer une clé de cache (non cryptographique)
    return hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()[:8]  # nosec B324


# ✅ S2: Fonction key_func pour rate limiting par utilisateur
# (si authentifié) ou par IP
# ✅ C2: Ajout du versioning pour invalider automatiquement les anciens rate limits
def get_rate_limit_key() -> str:
    """Génère une clé pour le rate limiting basée sur l'utilisateur
    (si authentifié) ou l'IP.

    ✅ S2: Rate limiting par utilisateur pour éviter qu'un utilisateur malveillant
    puisse contourner les limites en changeant d'IP.

    ✅ C2: Inclut un hash de configuration pour invalider automatiquement
    les anciens rate limits lors des changements de configuration.

    Returns:
        Clé de rate limiting
        (format: "v{hash}:user:{user_id}" ou "v{hash}:ip:{ip_address}")
    """
    # Récupérer le hash de version
    version_hash = get_rate_limit_config_hash()

    # Essayer de récupérer l'utilisateur authentifié
    try:
        user_public_id = get_jwt_identity()
        if user_public_id:
            # Utiliser l'ID utilisateur comme clé (plus sécurisé que l'IP seule)
            return f"v{version_hash}:user:{user_public_id}"
    except Exception:
        # Si l'authentification échoue, fallback sur IP
        pass

    # Fallback : utiliser l'adresse IP
    ip_address = get_remote_address()
    return f"v{version_hash}:ip:{ip_address}"


# ✅ S2: Rate limiting amélioré - limite par défaut réduite (5000 → 1000/heure)
# ✅ C2: Configuration dynamique depuis variables d'environnement
# ⚠️ C2 LOAD TESTING: Temporairement augmenté à 100000/hour pour tests de charge
_default_limit = os.getenv("RATELIMIT_DEFAULT_LIMITS", "100000 per hour")

# ✅ C2 Phase 2: TTL automatiques sur les clés de rate limit
# Durée de vie par défaut: 2 heures (7200 secondes)
# Les clés expirent automatiquement, évitant l'accumulation en mémoire
_rate_limit_ttl = int(os.getenv("RATELIMIT_KEY_TTL", "7200"))

# Créer le storage avec TTL automatiques si Redis est disponible
if limiter_storage != "memory://":
    _limiter_storage_with_ttl: str | RedisStorageWithTTL = RedisStorageWithTTL(
        limiter_storage, ttl_seconds=_rate_limit_ttl
    )
else:
    # Mode memory: pas besoin de TTL (les clés sont en mémoire volatile)
    _limiter_storage_with_ttl = limiter_storage

limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[_default_limit],
    storage_uri=_limiter_storage_with_ttl,
    strategy=os.getenv("RATELIMIT_STRATEGY", "moving-window"),
)

dispatch_status = {"is_running": False, "last_run_time": None}

# ✅ 2.7: Constantes pour DB Profiler
DB_PROFILER_N_PLUS_1_THRESHOLD = 10  # Seuil pour détecter pattern N+1
DB_PROFILER_WARNING_QUERY_COUNT = 15  # Seuil pour avertir trop de requêtes


# Event listener pour compter les requêtes SQL (performance metrics)
def setup_sql_event_listener():
    """Configure l'event listener SQLAlchemy pour compter les requêtes SQL."""
    from sqlalchemy import engine
    from sqlalchemy import event as sqlalchemy_event

    from services.unified_dispatch import performance_metrics

    def _receive_before_cursor_execute(
        _conn, _cursor, statement, _parameters, _context, _executemany
    ):
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

    sqlalchemy_event.listens_for(engine.Engine, "before_cursor_execute")(
        _receive_before_cursor_execute
    )


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
                    app.logger.error(
                        "[DB Profiler] 🚨 PATTERN N+1 détecté sur %s %s",
                        request.method,
                        request.path,
                    )

                # Ajouter headers de profiling si demandé
                if os.getenv("DB_PROFILING_HEADERS", "false").lower() == "true":
                    response.headers["X-DB-Query-Count"] = str(stats["query_count"])
                    response.headers["X-DB-Total-Time-Ms"] = str(stats["total_time_ms"])

            return response

        app.logger.info("[DB Profiler] Middleware de profiling configuré")
    else:
        app.logger.debug(
            "[DB Profiler] Profiling désactivé (set ENABLE_DB_PROFILING=true)"
        )


#  JWT error handlers
@jwt.expired_token_loader
def expired_token_callback(_jwt_header, _jwt_payload):
    return jsonify({"error": "Le token a expiré. Veuillez vous reconnecter."}), 401


@jwt.invalid_token_loader
def invalid_token_callback(error):
    """Callback pour tokens invalides (inclut audience invalide).

    ✅ S3: Détecte si une legacy key pourrait fonctionner (pour audit).

    Note: Flask-JWT-Extended ne permet pas de re-décoder avec une autre clé
    dans ce callback.
    Les tokens signés avec des legacy keys nécessiteront une reconnexion
    de l'utilisateur.
    Ce callback sert principalement à détecter et logger l'utilisation de legacy keys.

    Note: Les tokens avec audience invalide sont rejetés via check_if_token_revoked()
    et déclenchent ce callback avec une erreur générique.
    """
    # ✅ S3: Détecter si une legacy key pourrait fonctionner (pour audit uniquement)
    error_str = str(error) if error else ""
    if "signature" in error_str.lower() or "Invalid signature" in str(error):
        try:
            from flask import request

            from security.jwt_legacy_keys import try_decode_with_legacy_keys

            # Récupérer le token depuis le header Authorization
            auth_header = request.headers.get("Authorization", "")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1].strip()
                payload, used_key = try_decode_with_legacy_keys(token)

                if payload and used_key:
                    # Token décodé avec une legacy key - logger pour audit
                    # Note: Flask-JWT-Extended ne permet pas de "re-décoder"
                    # avec une autre clé dans ce callback. Le token sera rejeté
                    # et l'utilisateur devra se reconnecter.
                    msg = (
                        "[JWT Legacy] ✅ Token décodé avec legacy key "
                        "(user_id: %s, endpoint: %s)"
                    )
                    app_logger.info(
                        msg,
                        payload.get("sub") or payload.get("identity"),
                        request.path,
                    )
                    # Incrémenter métrique pour monitoring
                    try:
                        from security.security_metrics import (
                            security_token_invalidations_total,
                        )

                        security_token_invalidations_total.labels(
                            reason="legacy_key_detected"
                        ).inc()
                    except Exception:
                        pass  # Ne pas bloquer si métriques indisponibles
        except Exception as legacy_error:
            app_logger.debug(
                "[JWT Legacy] Erreur lors de la détection legacy keys: %s", legacy_error
            )

    # Le message d'erreur est générique pour ne pas révéler trop d'informations
    # Les détails spécifiques (audience invalide) sont dans les logs
    return jsonify({"error": "Token invalide. Veuillez vous reconnecter."}), 422


@jwt.unauthorized_loader
def missing_token_callback(_error):
    return jsonify(
        {"error": "Token d'accès manquant. Veuillez vous authentifier."}
    ), 401


@jwt.needs_fresh_token_loader
def token_not_fresh_callback(_jwt_header, _jwt_payload):
    return jsonify(
        {"error": "Le token n'est pas frais. Veuillez vous reconnecter."}
    ), 401


# ✅ Phase 3: Callback pour vérifier la blacklist des tokens
# ✅ SECURITY: Validation audience intégrée pour rejeter automatiquement
# les tokens avec audience invalide
@jwt.token_in_blocklist_loader
def check_if_token_revoked(_jwt_header, jwt_payload):
    """Vérifie si le token JWT est dans la blacklist et valide l'audience.

    Args:
        _jwt_header: En-tête JWT (non utilisé)
        jwt_payload: Payload JWT

    Returns:
        True si le token est révoqué (blacklisté) ou a une audience invalide,
        False sinon
    """
    from security.security_metrics import security_invalid_audience_total
    from security.token_blacklist import is_token_blacklisted

    # ✅ SECURITY: Valider audience AVANT de vérifier la blacklist
    audience_valid, reason = validate_jwt_audience(jwt_payload)
    if not audience_valid:
        app_logger.warning(
            "[JWT Security] Token avec audience invalide rejeté (raison: %s)", reason
        )
        # Incrémenter métrique Prometheus
        security_invalid_audience_total.labels(reason=reason).inc()
        return True  # Rejeter token avec audience invalide

    # Vérifier si le token est dans la blacklist
    # ✅ SECURITY: Flask-JWT-Extended génère automatiquement un jti
    # pour chaque token
    # Tous les tokens créés via create_access_token/create_refresh_token
    # ont un jti unique
    jti = jwt_payload.get("jti")
    if jti:
        return is_token_blacklisted(jti=jti)

    # ⚠️ FALLBACK: Si pas de jti (cas théorique, ne devrait pas arriver)
    # La blacklist utilisera le hash du token comme fallback
    # Ce cas ne devrait jamais se produire avec Flask-JWT-Extended moderne
    msg = (
        "[JWT Security] Token sans jti détecté dans check_if_token_revoked. "
        "Ce cas ne devrait pas se produire avec Flask-JWT-Extended moderne."
    )
    app_logger.warning(msg)
    # Ne pas vérifier la blacklist si pas de jti (le token sera accepté)
    # car on ne peut pas l'identifier de manière fiable sans jti
    return False


# ✅ Hardening JWT: Validation explicite de l'audience (défense en profondeur)
@jwt.additional_claims_loader
def add_claims_to_access_token(_identity):
    """Ajoute des claims supplémentaires au token
    (déjà fait dans auth.py, mais défense en profondeur).

    Args:
        _identity: Identité de l'utilisateur
        (non utilisé, mais requis par Flask-JWT-Extended)
    """
    # Les claims sont déjà ajoutés dans auth.py lors de la création du token
    # Ce callback est optionnel mais peut servir de vérification supplémentaire
    return {}


def validate_jwt_audience(jwt_payload: dict[str, Any]) -> tuple[bool, str]:
    """Valide que l'audience du token JWT correspond à l'audience attendue.

    Args:
        jwt_payload: Payload JWT décodé

    Returns:
        Tuple (is_valid, reason):
        - is_valid: True si l'audience est valide, False sinon
        - reason: Raison de l'échec ("missing" ou "wrong_audience") ou "valid" si OK
    """
    # ✅ Accepter les deux audiences : API standard et entreprise mobile
    valid_audiences = {"atmr-api", "atmr-mobile-enterprise"}
    token_audience = jwt_payload.get("aud")

    if not token_audience:
        app_logger.warning("[JWT Security] Token sans claim 'aud' (audience)")
        return False, "missing"

    if token_audience not in valid_audiences:
        app_logger.warning(
            "[JWT Security] Audience invalide: %s (attendues: %s)",
            token_audience,
            valid_audiences,
        )
        return False, "wrong_audience"

    return True, "valid"


#  Decorator role_required
def role_required(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            from models import User  # import local pour éviter les cycles

            user_public_id = get_jwt_identity()

            # ✅ Vérifier d'abord le claim du token JWT
            # (priorité pour les switchs de rôle)
            # Cela permet aux tokens créés lors d'un switch
            # (driver <-> company) de fonctionner
            # même si l'utilisateur dans la DB a un rôle différent
            token_role = None
            try:
                from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
                    get_jwt,
                )

                jwt_data = get_jwt()
                if jwt_data and "role" in jwt_data:
                    token_role = jwt_data["role"]
            except Exception:
                # Si on ne peut pas récupérer le claim,
                # continuer avec la vérification DB
                pass

            user = User.query.filter_by(public_id=user_public_id).first()

            if not user:
                app_logger.warning(
                    "Utilisateur non trouvé pour le public_id : %s", user_public_id
                )
                abort(404, description="Utilisateur non trouvé")

            # Convertir les rôles en objets UserRole pour la comparaison
            from models import UserRole

            allowed_roles = []

            # Gérer les deux formats :
            # @role_required(['ADMIN', 'COMPANY']) et
            # @role_required(UserRole.company)
            role_list: list[str] = []
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
                        app_logger.warning(
                            "Rôle invalide dans la configuration : %s",
                            role_str,
                        )

            # ✅ Vérifier d'abord le claim du token,
            # puis le rôle de l'utilisateur dans la DB
            # Cela permet aux tokens créés lors d'un switch de fonctionner
            role_check_passed = False
            if token_role:
                # Vérifier si le claim du token correspond
                # à un des rôles autorisés
                try:
                    token_user_role = UserRole[token_role.upper()]
                    if token_user_role in allowed_roles:
                        role_check_passed = True
                except KeyError:
                    # Le claim du token n'est pas un rôle valide,
                    # continuer avec la vérification DB
                    pass

            # Si le claim du token n'a pas passé,
            # vérifier le rôle de l'utilisateur dans la DB
            if not role_check_passed and user.role not in allowed_roles:
                warning_msg = (
                    "⛔ Accès refusé : %s (%s) a tenté d'accéder "
                    + "à une route restreinte."
                )
                app_logger.warning(warning_msg, user.username, user.role)
                abort(403, description="Accès non autorisé")

            return fn(*args, **kwargs)

        return wrapper

    return decorator
