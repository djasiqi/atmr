# backend/ext.py
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, get_jwt_identity
from flask_mail import Mail
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from typing import Literal, cast
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO
from functools import wraps
from flask import abort, jsonify
import logging
import os
import redis

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

# ⚠️ Ne fixe PAS CORS/path ici pour éviter les conflits — tout est défini dans app.py (socketio.init_app)
# Active la file Redis si disponible (scaling multi-workers).
# Typage strict de async_mode pour contenter Pylance/pyright.
AsyncMode = Literal["threading", "eventlet", "gevent", "gevent_uwsgi"]
_env_async = (os.getenv("SOCKETIO_ASYNC_MODE") or "eventlet").strip().lower()
_allowed_modes = {"threading", "eventlet", "gevent", "gevent_uwsgi"}
if _env_async not in _allowed_modes:
    # fallback sûr si une valeur inconnue est fournie
    _env_async = "eventlet"
ASYNC_MODE: AsyncMode = cast(AsyncMode, _env_async)

socketio = SocketIO(
    async_mode=ASYNC_MODE,
    message_queue=REDIS_URL if redis_client else None,
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["5000 per hour"],
    storage_uri=limiter_storage,
)

dispatch_status = {'is_running': False, 'last_run_time': None}
app_logger = logging.getLogger('app')


#  JWT error handlers 
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({"error": "Le token a expiré. Veuillez vous reconnecter."}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({"error": "Token invalide. Veuillez vous reconnecter."}), 422

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({"error": "Token d'accès manquant. Veuillez vous authentifier."}), 401

@jwt.needs_fresh_token_loader
def token_not_fresh_callback(jwt_header, jwt_payload):
    return jsonify({"error": "Le token n'est pas frais. Veuillez vous reconnecter."}), 401

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

            if user.role not in roles:
                app_logger.warning("⛔ Accès refusé : %s (%s) a tenté d'accéder à une route restreinte.",
                                   user.username, user.role)
                abort(403, description="Accès non autorisé")

            return fn(*args, **kwargs)
        return wrapper
    return decorator
