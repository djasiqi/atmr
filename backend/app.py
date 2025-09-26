# backend/app.py
import os
import json
import logging
from enum import Enum

from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

from flask import Flask, request, jsonify, make_response, send_from_directory, redirect
from flask_cors import CORS
from flask_talisman import Talisman
from sqlalchemy import event
from sqlalchemy.engine import Engine
from werkzeug.utils import safe_join
from werkzeug.exceptions import HTTPException, NotFound

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Extensions & config
from ext import db, jwt, mail, bcrypt, migrate, limiter, socketio
from config import config

from flask_socketio import join_room, leave_room, emit
from flask_jwt_extended import decode_token
from models import User, Driver


# 🔒 Hard-fail si secrets manquants (protège contre un démarrage "muet")
if not (os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY")):
    raise RuntimeError(
        "JWT_SECRET_KEY ou SECRET_KEY manquant(e). "
        "Ajoute-les dans backend/.env puis redémarre."
    )

# --- SQLite PRAGMA (clé étrangères ON) ---
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, _):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        # Ne pas crasher si ce n'est pas SQLite
        pass


# --- JSON encoder (legacy Flask <2.3) ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def create_app(config_name=os.getenv("FLASK_CONFIG", "development")):
    app = Flask(__name__)

    # Flask < 2.3 : encoder custom, sinon ignoré proprement
    try:
        app.json_encoder = CustomJSONEncoder
    except Exception:
        pass

    # 1) Config
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # 2) Extensions
    db.init_app(app)
    jwt.init_app(app)
    mail.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db)
    if app.config.get("RATELIMIT_ENABLED", True):
        limiter.init_app(app)
        
    # Initialize Celery
    try:
        from celery_app import init_app as init_celery
        init_celery(app)
        app.logger.info("Celery initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize Celery: {e}")

    # --- Socket.IO
    async_mode = os.getenv("SOCKETIO_ASYNC_MODE", "eventlet")

    if config_name == "development":
        cors_origins = "*"             # dev permissif
        sio_logger = True              # logs Engine.IO pour debug
        sio_engineio_logger = True
    else:
        cors_origins = os.getenv("SOCKETIO_CORS_ORIGINS", "").split(",") if os.getenv("SOCKETIO_CORS_ORIGINS") else []
        sio_logger = False
        sio_engineio_logger = False

    # En dev Android, on peut vouloir désactiver l’upgrade WebSocket (polling-only)
    # Active si SIO_DISABLE_UPGRADES=true (par défaut) et seulement en dev.
    allow_ws_upgrades = True
    if config_name == "development":
        if os.getenv("SIO_DISABLE_UPGRADES", "true").lower() == "true":
            allow_ws_upgrades = False

    socketio.init_app(
        app,
        async_mode=async_mode,
        cors_allowed_origins=cors_origins if cors_origins else [],
        # 🔗 Path unifié (sans slash final) — doit matcher Nginx/clients
        path="/socket.io",
        logger=sio_logger,
        engineio_logger=sio_engineio_logger,
        # ⏱️ Paramètres de robustesse raisonnables (dev/prod)
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=10e6,  # 10MB
        allow_upgrades=allow_ws_upgrades
    )

    # 🔎 Log léger et unique pour les requêtes Socket.IO (handshake / polling / ws)
    @app.before_request
    def _log_socketio_requests():
        p = (request.path or "")
        if p.startswith("/socket.io"):
            app.logger.debug(
                "📡 SIO %s %s from %s", request.method, request.full_path, request.remote_addr
            )
        return None



    # 3) Uploads: config + route statique
    uploads_root = os.path.join(app.root_path, "uploads")
    os.makedirs(uploads_root, exist_ok=True)
    app.config.setdefault("UPLOADS_DIR", uploads_root)
    app.config.setdefault("UPLOADS_PUBLIC_BASE", "/uploads")

    @app.route("/uploads/<path:filename>")
    def serve_uploads(filename):
        full_path = safe_join(app.config["UPLOADS_DIR"], filename)
        if not full_path or not os.path.isfile(full_path):
            raise NotFound()
        directory, fname = os.path.split(full_path)
        # conditional=True => ETag/If-Modified-Since OK (cache navigateur)
        return send_from_directory(directory, fname, conditional=True)

    # 4) Sécurité (CSP compatible ws/wss pour Socket.IO)
    if config_name in {"development", "testing"}:
        csp = {
            "default-src": "'self'",
            "script-src": "'self'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: blob:",
            "connect-src": "'self' http://localhost:3000 http://127.0.0.1:3000 ws: wss:",
        }
        force_https = False   # <— ne redirige PAS en dev, sinon /socket.io/ casse
    else:
        frontend_url = os.getenv("FRONTEND_URL", "https://ton-frontend.tld")
        csp = {
            "default-src": "'self'",
            "script-src": "'self'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: blob:",
            "connect-src": f"'self' {frontend_url} ws: wss:",
        }
        force_https = True

    talisman = Talisman(content_security_policy=csp, force_https=force_https)
    talisman.init_app(app)


    # (Option) Désactiver CSP pour les réponses JSON (API)
    @app.after_request
    def strip_csp_for_json(resp):
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in ct:
            resp.headers.pop("Content-Security-Policy", None)
            resp.headers.pop("Content-Security-Policy-Report-Only", None)
        return resp

    # 5) CORS (liste d'origines explicite)
    CORS(
        app,
        resources={r"/*": {"origins": "*" if cors_origins == "*" else cors_origins}},
        supports_credentials=True,
        expose_headers=["Content-Type", "Authorization"],
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    # 6) Sentry (désactivé en tests)
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn and config_name != "testing":
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[FlaskIntegration()],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
            environment=config_name,
        )

    # 7) Logging : niveau global + silencieux pour werkzeug
    app_log_level = getattr(logging, os.getenv("APP_LOG_LEVEL", "INFO").upper(), logging.INFO)
    app.logger.setLevel(app_log_level)
    logging.getLogger("werkzeug").setLevel(
        getattr(logging, os.getenv("WERKZEUG_LOG_LEVEL", "ERROR").upper(), logging.ERROR)
    )

    # 8) Injecter l’app dans la queue unified_dispatch pour les threads
    from services.unified_dispatch import queue as ud_queue
    ud_queue.init_app(app)

    # 9) Routes / sockets / handlers
    with app.app_context():
        from sqlalchemy.orm import configure_mappers
        configure_mappers()

        from routes_api import init_namespaces
        init_namespaces(app)

        # --- Compatibilité: alias legacy sans /api vers /api/... ---
        # Permet au front existant (ex: /auth/login) de continuer à fonctionner pendant la transition.
        LEGACY_PREFIXES = (
            "auth", "clients", "admin", "companies", "driver",
            "bookings", "payments", "utils", "messages",
            "company_dispatch", "geocode", "medical", "ai",
        )

        @app.before_request
        def legacy_api_shim():
            path = request.path or "/"
            # Laisser passer les routes non-API utiles
            if (
                path.startswith("/api/")
                or path.startswith("/uploads/")
                or path.startswith("/socket.io")  # ⚠️ ne jamais toucher au transport SIO
                or path in {"/", "/health", "/config"}
            ):
                return None
            # Si le chemin commence par un prefix legacy connu → shim
            if any(path == f"/{p}" or path.startswith(f"/{p}/") for p in LEGACY_PREFIXES):
                # Preflight CORS: renvoyer 204 "OK"
                if request.method == "OPTIONS":
                    return make_response("", 204)
                # Autres méthodes: redirection 307 (préserve méthode + body)
                return redirect(f"/api{path}", code=307)
            return None

        from sockets.chat import init_chat_socket
        init_chat_socket(socketio)

        @app.route("/")
        def index():
            return jsonify({"message": "Welcome to the Transport API"}), 200

        @app.route("/health")
        def health():
            return jsonify({"status": "ok"}), 200

        @app.route("/config")
        def show_config():
            return jsonify(
                {
                    "env": config_name,
                    "DATABASE_URI": app.config.get("SQLALCHEMY_DATABASE_URI"),
                    "UPLOADS_PUBLIC_BASE": app.config.get("UPLOADS_PUBLIC_BASE"),
                }
            ), 200


        # Réduction du bruit de logs pour certains endpoints “polling”
        NOISY_PATHS = {
            "/companies/me/dispatch/status",
            "/company_dispatch/status",
        }

        @app.after_request
        def log_request_info(response):
            if request.path in NOISY_PATHS and request.method in ("GET", "OPTIONS"):
                return response
            app.logger.debug("%s %s -> %s", request.method, request.path, response.status_code)
            return response

        # JWT : handlers d'erreurs
        @jwt.expired_token_loader
        def expired_token_callback(jwt_header, jwt_payload):
            return jsonify({"error": "token_expired", "message": "Signature has expired"}), 401

        @jwt.invalid_token_loader
        def invalid_token_callback(error):
            return jsonify({"error": "invalid_token", "message": str(error)}), 422

        @jwt.unauthorized_loader
        def missing_token_callback(error):
            return jsonify({"error": "missing_token", "message": str(error)}), 401

        # Laisser les HTTPException (404, 403, ...) passer sans être transformées en 500
        @app.errorhandler(HTTPException)
        def handle_http_exception(e):
            if isinstance(e, NotFound):
                app.logger.warning("404 on path: %s", request.path)
            return jsonify({"error": e.name, "message": e.description}), e.code

        # Handler global pour les autres exceptions
        @app.errorhandler(Exception)
        def handle_exception(e):
            app.logger.error("Unhandled server error", exc_info=True)
            msg = str(e) if app.config.get("DEBUG") else "Une erreur interne est survenue."
            return jsonify({"error": "server_error", "message": msg}), 500

    # --- Rooms & connexions ---
    _sid_index = {}

    @socketio.on("disconnect")
    def handle_disconnect():
        info = _sid_index.pop(request.sid, None)
        print(f"👋 SIO disconnect sid={request.sid} info={info}")


    return app
