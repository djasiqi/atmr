# backend/app.py
# pyright: reportImportCycles = false

import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

# TypeAlias import (compatible Python 3.10+)
try:
    from typing import TypeAlias, override
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias
    
    # Import override if available  
    try:
        from typing_extensions import override  # type: ignore[assignment]
    except ImportError:
        # Fallback si override n'est pas disponible
        def override(func: Any) -> Any:
            return func

# --- Imports de libs tiers (tous en haut pour Ruff E402) ---
import sentry_sdk
from dotenv import load_dotenv
from flask import Flask, current_app, jsonify, make_response, request, send_from_directory
from flask_cors import CORS
from flask_talisman import Talisman
from sentry_sdk.integrations.flask import FlaskIntegration
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.middleware.proxy_fix import ProxyFix

# Extensions & config
from config import config
from ext import bcrypt, db, jwt, limiter, mail, migrate, socketio

# ---------- Chargement .env ----------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

# ---------- Types ----------
AsyncMode: TypeAlias = Literal["threading", "eventlet", "gevent", "gevent_uwsgi"]

# ---------- Garde-fou secrets ----------
if not (os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY")):
    msg = "JWT_SECRET_KEY ou SECRET_KEY manquant(e). Ajoute-les dans backend/.env puis redémarre."
    raise RuntimeError(
        msg
    )

# ---------- JSON encoder/provider ----------
class CustomJSONEncoder(json.JSONEncoder):
    @override
    def default(self, o):  # pyright: ignore[reportImplicitOverride]
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


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
            @override
            def default(self, o):
                if isinstance(o, Enum):
                    return o.value
                return super().default(o)

        app.json = CustomJSONProvider(app)
    except Exception:
        app.json_encoder = CustomJSONEncoder

    # 1) Config
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # ✅ Force UTF-8 encoding pour JSON et réponses
    app.config["JSON_AS_ASCII"] = False
    app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"

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
        app.logger.error("Failed to initialize Celery: %s", e)

    # --- Socket.IO ---
    app.logger.info("🔧 [INIT] Configuration Socket.IO...")
    allowed_modes: set[str] = {"threading", "eventlet", "gevent", "gevent_uwsgi"}
    env_mode = os.getenv("SOCKETIO_ASYNC_MODE", "eventlet")
    async_mode: AsyncMode = cast("AsyncMode", env_mode if env_mode in allowed_modes else "eventlet")

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
    if config_name == "development" and os.getenv("SIO_DISABLE_UPGRADES", "true").lower() == "true":
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
    app.logger.info(
        "✅ Socket.IO initialisé: async_mode=%s, cors=%s, allow_upgrades=%s",
        async_mode,
        "*" if cors_origins == "*" else "restricted",
        allow_ws_upgrades
    )

    # Log léger pour les requêtes Socket.IO
    @app.before_request
    def _log_socketio_requests():  # pyright: ignore[reportUnusedFunction]
        p = (request.path or "")
        if p.startswith("/socket.io"):
            app.logger.debug("📡 SIO %s %s from %s", request.method, request.full_path, request.remote_addr)

    # à l'initialisation Flask
    @app.teardown_appcontext
    def shutdown_session(exception=None):  # pyright: ignore[reportUnusedFunction]  # noqa: ARG001
        db.session.remove()


    # Gestion d'erreurs réseau (Bad file descriptor, etc.)
    # Constantes pour les codes d'erreur système
    BAD_FILE_DESCRIPTOR = 9
    
    @app.errorhandler(OSError)
    def handle_os_error(e: OSError):  # pyright: ignore[reportUnusedFunction]
        err = getattr(e, "errno", None)
        if err == BAD_FILE_DESCRIPTOR:  # Bad file descriptor
            app.logger.warning("Socket error (Bad file descriptor) - connection may have been closed")
            return jsonify({"error": "Connection closed"}), 499
        raise e

    # 3) Uploads
    uploads_root = Path(app.root_path, "uploads")
    Path(uploads_root, exist_ok=True).mkdir(parents=True, exist_ok=True)
    app.config.setdefault("UPLOADS_DIR", uploads_root)
    app.config.setdefault("UPLOADS_PUBLIC_BASE", "/uploads")

    @app.route("/uploads/<path:filename>")
    def serve_uploads(filename: str): # pyright: ignore[reportUnusedFunction]
        base = Path(app.config["UPLOADS_DIR"]).resolve()
        candidate = (base / filename).resolve()
        # anti-path traversal
        if not str(candidate).startswith(str(base)):
            raise NotFound
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
            "connect-src": "'self' http://localhost:3000 http://127.00.1:3000 ws: wss:",
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

    # Retirer CSP pour les réponses JSON et forcer UTF-8
    @app.after_request
    def strip_csp_for_json(resp):  # pyright: ignore[reportUnusedFunction]
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in ct:
            resp.headers.pop("Content-Security-Policy", None)
            resp.headers.pop("Content-Security-Policy-Report-Only", None)
            # ✅ Force UTF-8 encoding dans Content-Type (toujours)
            resp.headers["Content-Type"] = "application/json; charset=utf-8"
        # ✅ Force UTF-8 pour les réponses texte également
        elif "text/" in ct and "charset" not in ct:
            resp.headers["Content-Type"] = f"{resp.headers.get('Content-Type')}; charset=utf-8"
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

    # ✅ Ajout filtre PII si activé
    if os.getenv("MASK_PII_LOGS", "true").lower() == "true":
        from shared.logging_utils import PIIFilter
        pii_filter = PIIFilter()
        app.logger.addFilter(pii_filter)
        logging.getLogger("werkzeug").addFilter(pii_filter)
        logging.getLogger().addFilter(pii_filter)

    # 8) Unified dispatch queue
    from services.unified_dispatch import queue as ud_queue
    ud_queue.init_app(app)

    # 9) Routes / sockets / handlers
    app.logger.info("🔧 [INIT] Enregistrement des routes et handlers Socket.IO...")
    with app.app_context():
        from sqlalchemy.orm import configure_mappers

        import models
        _ = models  # Force l'import pour enregistrer les mappers

        configure_mappers()

        from routes_api import init_namespaces

        init_namespaces(app)

        # ✅ Enhanced healthcheck with DB/Redis checks
        from routes.feature_flags_routes import feature_flags_bp
        from routes.healthcheck import healthcheck_bp
        from routes.ml_monitoring import ml_monitoring_bp
        from routes.proactive_alerts import register_proactive_alerts_routes

        app.register_blueprint(healthcheck_bp)
        app.register_blueprint(feature_flags_bp)
        app.register_blueprint(ml_monitoring_bp)

        # ✅ Enregistrer les routes d'alertes proactives
        register_proactive_alerts_routes(app)

        # Réponse générique aux préflights CORS (toutes routes)
        @app.before_request
        def _cors_preflights_any():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            return None

        # Shim: réécrit /v<number>/* -> /api/* (supprime la version explicite)
        @app.before_request
        def _strip_top_version_prefix():  # pyright: ignore[reportUnusedFunction]
            # Exemple: /v1/companies/me -> /api/companies/me
            p = request.path or ""
            if re.match(r"^/v\d+(?:/.*)?$", p):
                environ = request.environ
                new_path = re.sub(r"^/v\d+", "/api", p)
                environ["ORIGINAL_PATH_INFO_VERSION"] = p
                environ["PATH_INFO"] = new_path
                current_app.logger.info(
                    "Version shim internal reroute %s %s -> %s", request.method, p, new_path
                )
                return
            return

        # Shim générique pour /api[/vX]/auth/login si RESTX rate
        @app.before_request
        def _auth_login_shim_any_version():  # pyright: ignore[reportUnusedFunction]
            p = request.path or ""
            if request.method in ("POST", "OPTIONS") and re.fullmatch(r"/api(?:/v\d+)?/auth/login", p):
                # Correspond à /api/auth/login ou /api/v<number>/auth/login
                if request.method == "OPTIONS":
                    return make_response("", 204)
                try:
                    from routes.auth import Login
                    return Login().post()
                except Exception:
                        # Laisse la suite normale (404 si absent)
                        return None
            return None

        # Compat: routes legacy sans /api
        legacy_prefixes = (
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
        def legacy_api_shim():  # pyright: ignore[reportUnusedFunction]
            path = request.path or "/"
            if path == "/api" or path.startswith("/api/"):
                return None
            if (
                path in {"/", "/health", "/config", "/docs", "/favicon.ico", "/robots.txt"} or path.startswith(("/swaggerui/", "/static/", "/uploads/", "/socket.io"))
            ):
                return None

            current_app.logger.debug("Legacy shim: %s %s", request.method, path)

            if any(path == f"/{p}" or path.startswith(f"/{p}/") for p in legacy_prefixes):
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

        app.logger.info("🔧 [INIT] Enregistrement des handlers Socket.IO chat...")
        from sockets.chat import init_chat_socket

        init_chat_socket(socketio)
        app.logger.info("✅ Handlers Socket.IO chat enregistrés")

        @app.route("/")
        def index():  # pyright: ignore[reportUnusedFunction]
            return jsonify({"message": "Welcome to the Transport API"}), 200

        # Compat: accès direct à l'autocomplete (certains environnements/proxy peuvent rater la déclaration RESTX)
        @app.route("/api/geocode/autocomplete", methods=["GET", "OPTIONS"])
        def _compat_geocode_autocomplete():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            try:
                from routes.geocode import GeocodeAutocomplete
                res = GeocodeAutocomplete()
                return res.get()
            except Exception as err:
                # Laisse la 404 standard si la ressource n'est pas dispo
                raise NotFound from err

        # Compat: accès direct au login (certains environnements/proxy peuvent rater la déclaration RESTX)
        from routes.auth import Login  # Import au niveau module
        
        @app.route("/api/auth/login", methods=["POST", "OPTIONS"])
        def _compat_auth_login():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            # Laisse passer les réponses normales (200/4xx)
            return Login().post()

        @app.route("/auth/login", methods=["POST", "OPTIONS"])
        def _compat_auth_login_root():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            return Login().post()

        @app.route("/api/v<int:version>/auth/login", methods=["POST", "OPTIONS"])
        def _compat_auth_login_v(version: int):  # pyright: ignore[reportUnusedFunction]  # noqa: ARG001
            if request.method == "OPTIONS":
                return make_response("", 204)
            return Login().post()

        # --- Compat: endpoints companies les plus utilisés (évite 404 si RESTX rate) ---
        from routes.companies import CompanyDriversList, CompanyMe
        
        @app.route("/api/companies/me", methods=["GET", "PUT", "OPTIONS"])
        def _compat_companies_me():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            res = CompanyMe()
            if request.method == "GET":
                return res.get()
            if request.method == "PUT":
                return res.put()
            raise NotFound

        @app.route("/api/v<int:version>/companies/me", methods=["GET", "PUT", "OPTIONS"])
        def _compat_companies_me_v(version: int):  # pyright: ignore[reportUnusedFunction]  # noqa: ARG001
            if request.method == "OPTIONS":
                return make_response("", 204)
            res = CompanyMe()
            if request.method == "GET":
                return res.get()
            if request.method == "PUT":
                return res.put()
            raise NotFound

        @app.route("/api/companies/me/drivers", methods=["GET", "OPTIONS"])
        def _compat_companies_me_drivers():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            return CompanyDriversList().get()

        @app.route("/api/v<int:version>/companies/me/drivers", methods=["GET", "OPTIONS"])
        def _compat_companies_me_drivers_v(version: int):  # pyright: ignore[reportUnusedFunction]  # noqa: ARG001
            if request.method == "OPTIONS":
                return make_response("", 204)
            return CompanyDriversList().get()

        @app.route("/health")
        def health():  # pyright: ignore[reportUnusedFunction]
            return jsonify({"status": "ok"}), 200

        @app.route("/config")
        def show_config():  # pyright: ignore[reportUnusedFunction]
            return jsonify(
                {
                    "env": config_name,
                    "DATABASE_URI": app.config.get("SQLALCHEMY_DATABASE_URI"),
                    "UPLOADS_PUBLIC_BASE": app.config.get("UPLOADS_PUBLIC_BASE"),
                }
            ), 200

        noisy_paths = {
            "/companies/me/dispatch/status",
            "/company_dispatch/status",
        }

        @app.after_request
        def log_request_info(response):  # pyright: ignore[reportUnusedFunction]
            if request.path in noisy_paths and request.method in ("GET", "OPTIONS"):
                return response
            app.logger.debug("%s %s -> %s", request.method, request.path, response.status_code)
            return response

        # JWT : handlers d'erreurs
        @jwt.expired_token_loader
        def expired_token_callback(jwt_header, jwt_payload):  # pyright: ignore[reportUnusedFunction]  # noqa: ARG001
            return jsonify({"error": "token_expired", "message": "Signature has expired"}), 401

        @jwt.invalid_token_loader
        def invalid_token_callback(error):  # pyright: ignore[reportUnusedFunction]
            return jsonify({"error": "invalid_token", "message": str(error)}), 422

        @jwt.unauthorized_loader
        def missing_token_callback(error):  # pyright: ignore[reportUnusedFunction]
            return jsonify({"error": "missing_token", "message": str(error)}), 401

        @app.errorhandler(HTTPException)
        def handle_http_exception(e: HTTPException):  # pyright: ignore[reportUnusedFunction]
            if isinstance(e, NotFound):
                app.logger.warning("404 on path: %s", request.path)
            status_code: int = int(e.code or 500)  # <- évite int | None
            return jsonify({"error": e.name, "message": e.description}), status_code

        @app.errorhandler(Exception)
        def handle_exception(e: Exception):  # pyright: ignore[reportUnusedFunction]
            app.logger.exception("Unhandled server error")
            msg = str(e) if app.config.get("DEBUG") else "Une erreur interne est survenue."
            return jsonify({"error": "server_error", "message": msg}), 500

    # Note: handler disconnect géré dans sockets/chat.py (pas de doublon)

    return app
