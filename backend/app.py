# backend/app.py

import os
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Literal, cast

# TypeAlias (Py<3.10 via typing_extensions)
try:
    from typing import TypeAlias  # type: ignore
except Exception:  # pragma: no cover
    from typing_extensions import TypeAlias  # type: ignore

# --- Imports de libs tiers (tous en haut pour Ruff E402) ---
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response, send_from_directory, current_app
from flask_cors import CORS
from flask_talisman import Talisman
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.middleware.proxy_fix import ProxyFix

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Extensions & config
from ext import db, jwt, mail, bcrypt, migrate, limiter, socketio
from config import config

# ---------- Chargement .env ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# ---------- Types ----------
AsyncMode: TypeAlias = Literal["threading", "eventlet", "gevent", "gevent_uwsgi"]

# ---------- Garde-fou secrets ----------
if not (os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY")):
    raise RuntimeError(
        "JWT_SECRET_KEY ou SECRET_KEY manquant(e). Ajoute-les dans backend/.env puis redémarre."
    )

# ---------- JSON encoder/provider ----------
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def create_app(config_name: str | None = None):
    if config_name is None:
        config_name = os.getenv("FLASK_CONFIG", "development")

    app = Flask(__name__)

    # Désactiver les slashes stricts pour éviter les redirections 308
    app.url_map.strict_slashes = False
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    if os.getenv("FLASK_ENV", "production") != "production":
        app.config["PREFERRED_URL_SCHEME"] = "http"
        app.config["SESSION_COOKIE_SECURE"] = False

    # Flask 2.3+ : JSON Provider ; fallback <2.3 : json_encoder
    try:
        from flask.json.provider import DefaultJSONProvider

        class CustomJSONProvider(DefaultJSONProvider):
            def default(self, obj):  # type: ignore[override]
                if isinstance(obj, Enum):
                    return obj.value
                return super().default(obj)

        app.json = CustomJSONProvider(app)  # type: ignore[assignment]
    except Exception:
        app.json_encoder = CustomJSONEncoder  # type: ignore[attr-defined, assignment]

    # 1) Config
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # 2) Extensions
    db.init_app(app)
    jwt.init_app(app)
    mail.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db, compare_type=True, render_as_batch=True)

    if app.config.get("RATELIMIT_ENABLED", True):
        limiter.init_app(app)

    # Celery (optionnel)
    try:
        from celery_app import init_app as init_celery
        init_celery(app)
        app.logger.info("Celery initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize Celery: {e}")

    # --- Socket.IO ---
    allowed_modes: set[str] = {"threading", "eventlet", "gevent", "gevent_uwsgi"}
    env_mode = os.getenv("SOCKETIO_ASYNC_MODE", "eventlet")
    async_mode: AsyncMode = cast(AsyncMode, env_mode if env_mode in allowed_modes else "eventlet")

    if config_name == "development":
        cors_origins: str | list[str] = "*"  # dev permissif
        sio_logger = True
        sio_engineio_logger = True
    else:
        cors_origins = (
            os.getenv("SOCKETIO_CORS_ORIGINS", "").split(",")
            if os.getenv("SOCKETIO_CORS_ORIGINS")
            else []
        )
        sio_logger = False
        sio_engineio_logger = False

    allow_ws_upgrades = True
    if config_name == "development":
        if os.getenv("SIO_DISABLE_UPGRADES", "true").lower() == "true":
            allow_ws_upgrades = False

    # NB: pas de 'upgrade_timeout' (paramètre inexistant) ni 'cookie=True' (type incompatible)
    socketio.init_app(
        app,
        async_mode=async_mode,
        cors_allowed_origins=cors_origins if cors_origins else [],
        path="/socket.io",
        logger=sio_logger,
        engineio_logger=sio_engineio_logger,
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=10_000_000,  # int
        allow_upgrades=allow_ws_upgrades,
        cors_credentials=True,
    )

    # Log léger pour les requêtes Socket.IO
    @app.before_request
    def _log_socketio_requests():
        p = (request.path or "")
        if p.startswith("/socket.io"):
            app.logger.debug("📡 SIO %s %s from %s", request.method, request.full_path, request.remote_addr)
        return None
    
    # à l’initialisation Flask
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db.session.remove()


    # Gestion d'erreurs réseau (Bad file descriptor, etc.)
    @app.errorhandler(OSError)
    def handle_os_error(e: OSError):
        err = getattr(e, "errno", None)
        if err == 9:  # Bad file descriptor
            app.logger.warning("Socket error (Bad file descriptor) - connection may have been closed")
            return jsonify({"error": "Connection closed"}), 499
        raise e

    # 3) Uploads
    uploads_root = os.path.join(app.root_path, "uploads")
    os.makedirs(uploads_root, exist_ok=True)
    app.config.setdefault("UPLOADS_DIR", uploads_root)
    app.config.setdefault("UPLOADS_PUBLIC_BASE", "/uploads")

    @app.route("/uploads/<path:filename>")
    def serve_uploads(filename: str):
        base = Path(app.config["UPLOADS_DIR"]).resolve()
        candidate = (base / filename).resolve()
        # anti-path traversal
        if not str(candidate).startswith(str(base)):
            raise NotFound()
        directory = str(base)
        fname = str(candidate.relative_to(base)).replace("\\", "/")
        return send_from_directory(directory, fname, conditional=True)

    # 4) Sécurité (CSP)
    if config_name in {"development", "testing"}:
        csp = {
            "default-src": "'self'",
            "script-src": "'self'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: blob:",
            "connect-src": "'self' http://localhost:3000 http://127.0.0.1:3000 ws: wss:",
        }
        force_https = False
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

    # Retirer CSP pour les réponses JSON
    @app.after_request
    def strip_csp_for_json(resp):
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in ct:
            resp.headers.pop("Content-Security-Policy", None)
            resp.headers.pop("Content-Security-Policy-Report-Only", None)
        return resp

    # 5) CORS
    CORS(
        app,
        resources={r"/*": {"origins": "*" if cors_origins == "*" else cors_origins}},
        supports_credentials=True,
        expose_headers=["Content-Type", "Authorization"],
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    # 6) Sentry
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn and config_name != "testing":
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[FlaskIntegration()],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
            environment=config_name,
        )

    # 7) Logging
    app_log_level = getattr(logging, os.getenv("APP_LOG_LEVEL", "INFO").upper(), logging.INFO)
    app.logger.setLevel(app_log_level)
    logging.getLogger("werkzeug").setLevel(
        getattr(logging, os.getenv("WERKZEUG_LOG_LEVEL", "ERROR").upper(), logging.ERROR)
    )

    # 8) Unified dispatch queue
    from services.unified_dispatch import queue as ud_queue
    ud_queue.init_app(app)

    # 9) Routes / sockets / handlers
    with app.app_context():
        import models  # noqa: F401
        from sqlalchemy.orm import configure_mappers

        configure_mappers()

        from routes_api import init_namespaces

        init_namespaces(app)

        # Compat: routes legacy sans /api
        LEGACY_PREFIXES = (
            "auth",
            "clients",
            "admin",
            "companies",
            "driver",
            "bookings",
            "payments",
            "utils",
            "messages",
            "company_dispatch",
            "geocode",
            "medical",
            "ai",
            "routes",
            "dispatch",
            "tasks",
            "users",
        )

        @app.before_request
        def legacy_api_shim():
            path = request.path or "/"
            if path == "/api" or path.startswith("/api/"):
                return None
            if (
                path in {"/", "/health", "/config", "/docs", "/favicon.ico", "/robots.txt"}
                or path.startswith("/swaggerui/")
                or path.startswith("/static/")
                or path.startswith("/uploads/")
                or path.startswith("/socket.io")
            ):
                return None

            current_app.logger.debug("Legacy shim: %s %s", request.method, path)

            if any(path == f"/{p}" or path.startswith(f"/{p}/") for p in LEGACY_PREFIXES):
                if request.method == "OPTIONS":
                    return make_response("", 204)
                new_path = "/api" + path
                environ = request.environ
                environ["ORIGINAL_PATH_INFO"] = path
                environ["PATH_INFO"] = new_path
                current_app.logger.info(
                    "Legacy shim internal reroute %s %s -> %s", request.method, path, new_path
                )
                return None
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

        @app.errorhandler(HTTPException)
        def handle_http_exception(e: HTTPException):
            if isinstance(e, NotFound):
                app.logger.warning("404 on path: %s", request.path)
            status_code: int = int(e.code or 500)  # <- évite int | None
            return jsonify({"error": e.name, "message": e.description}), status_code

        @app.errorhandler(Exception)
        def handle_exception(e: Exception):
            app.logger.error("Unhandled server error", exc_info=True)
            msg = str(e) if app.config.get("DEBUG") else "Une erreur interne est survenue."
            return jsonify({"error": "server_error", "message": msg}), 500

    # --- Rooms & connexions ---
    _sid_index: dict[str, dict] = {}

    @socketio.on("disconnect")
    def handle_disconnect():
        # 'request.sid' est injecté par Flask-SocketIO → on garde une garde
        sid_val = getattr(request, "sid", None)
        if isinstance(sid_val, str):
            info = _sid_index.pop(sid_val, None)
            print(f"👋 SIO disconnect sid={sid_val} info={info}")
        else:
            print("👋 SIO disconnect (sid non disponible)")

    return app
