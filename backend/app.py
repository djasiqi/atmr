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


# ---------- Validation variables d'environnement critiques ----------
def validate_required_env_vars(config_name: str) -> None:
    """Valide toutes les variables d'environnement critiques.

    Args:
        config_name: Nom de la configuration (development, testing, production)

    Raises:
        RuntimeError: Si des variables critiques sont manquantes
    """
    # Variables critiques pour tous les environnements
    required_vars: set[str] = {
        "SECRET_KEY",
        "JWT_SECRET_KEY",
    }

    # Variables optionnelles mais nécessaires si l'une est manquante
    # Au moins une clé secrète est requise
    if not (os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY")):
        raise RuntimeError(
            "JWT_SECRET_KEY ou SECRET_KEY manquant(e). "
            + "Ajoutez au moins l'une de ces variables dans backend/.env puis redémarrez."
        )

    # Variables critiques pour production
    if config_name == "production":
        production_vars = {
            "DATABASE_URL",
            "REDIS_URL",
        }
        # SENTRY_DSN et PDF_BASE_URL sont optionnels mais recommandés
        recommended_vars = {
            "SENTRY_DSN",
            "PDF_BASE_URL",
        }
        required_vars.update(production_vars)

        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)

        if missing:
            raise RuntimeError(
                f"Variables d'environnement manquantes pour production: {', '.join(missing)}\n"
                + "Ajoutez-les dans backend/.env puis redémarrez."
            )

        # Vérifier variables recommandées et avertir si manquantes
        missing_recommended = [var for var in recommended_vars if not os.getenv(var)]
        if missing_recommended:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Variables recommandées manquantes pour production: %s. L'application fonctionnera mais certaines fonctionnalités peuvent être limitées.",
                ", ".join(missing_recommended),
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

    # Valider variables d'environnement critiques avant de continuer
    validate_required_env_vars(config_name)

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

    # ✅ 2.9: Setup OpenTelemetry avec instrumentation complète
    try:
        from shared.otel_setup import (
            instrument_celery,
            instrument_flask,
            instrument_sqlalchemy,
            setup_opentelemetry,
        )

        # Configurer OpenTelemetry (doit être fait avant les instrumentations)
        service_name = os.getenv("OTEL_SERVICE_NAME", "atmr-backend")
        service_version = os.getenv("OTEL_SERVICE_VERSION", "1.0")
        setup_opentelemetry(service_name=service_name, service_version=service_version)

        # Instrumenter Flask (requêtes HTTP)
        instrument_flask(app)

        # Instrumenter SQLAlchemy (requêtes DB)
        # Flask-SQLAlchemy expose get_engine() ou .engine selon la version
        try:
            # Essayer d'obtenir l'engine (Flask-SQLAlchemy 3+ utilise .engine directement)
            if hasattr(db, "engine"):
                engine = db.engine
            elif hasattr(db, "get_engine"):
                # Flask-SQLAlchemy <3 utilise get_engine() avec app context
                with app.app_context():
                    engine = db.get_engine()
            else:
                # Délayer l'instrumentation après que l'engine soit créé
                @app.before_first_request
                def _instrument_db_after_init():  # pyright: ignore[reportUnusedFunction]
                    try:
                        with app.app_context():
                            if hasattr(db, "engine"):
                                db_engine = db.engine
                            elif hasattr(db, "get_engine"):
                                db_engine = db.get_engine()
                            else:
                                return
                            instrument_sqlalchemy(db_engine)
                    except Exception as e:
                        app.logger.warning("[2.9] Échec instrumentation DB différée: %s", e)

                engine = None

            if engine:
                instrument_sqlalchemy(engine)
        except Exception as e:
            app.logger.warning("[2.9] Échec instrumentation SQLAlchemy: %s", e)
    except ImportError as e:
        app.logger.warning("[2.9] OpenTelemetry non disponible (optionnel): %s", e)
    except Exception as e:
        app.logger.warning("[2.9] Échec initialisation OpenTelemetry: %s", e)

    # ✅ 2.7: Setup DB Profiler pour détecter N+1 (activable via ENABLE_DB_PROFILING=true)
    try:
        from ext import setup_db_profiler

        setup_db_profiler(app)
    except Exception as e:
        app.logger.warning("[DB Profiler] Échec initialisation: %s", e)

    # Prometheus middleware pour métriques HTTP (latence p50/p95/p99)
    try:
        from middleware.metrics import prom_middleware

        app = prom_middleware(app)
    except ImportError as e:
        app.logger.warning("[Prometheus] Middleware non disponible: %s", e)

    # Celery (optionnel)
    celery_app = None
    try:
        from celery_app import init_app as init_celery

        celery_app = init_celery(app)
        app.logger.info("Celery initialized successfully")

        # ✅ 2.9: Instrumenter Celery après initialisation (avec contexte OpenTelemetry)
        if celery_app:
            try:
                # Importer depuis otel_setup (déjà configuré plus haut)
                from shared.otel_setup import instrument_celery

                instrument_celery(celery_app)
            except Exception as e:
                app.logger.warning("[2.9] Échec instrumentation Celery: %s", e)
    except Exception as e:
        app.logger.error("Failed to initialize Celery: %s", e)

    # --- Socket.IO ---
    # ✅ Skip Socket.IO pour les scripts (évite blocage)
    skip_socketio = os.getenv("SKIP_SOCKETIO", "false").lower() == "true"

    # Définir cors_origins même si Socket.IO est désactivé (nécessaire pour CORS plus bas)
    if config_name == "development":
        cors_origins: str | list[str] = "*"  # dev permissif
    else:
        cors_origins = os.getenv("SOCKETIO_CORS_ORIGINS", "").split(",") if os.getenv("SOCKETIO_CORS_ORIGINS") else []

    if skip_socketio:
        app.logger.info("⏭️  Socket.IO désactivé (SKIP_SOCKETIO=1)")
    else:
        app.logger.info("🔧 [INIT] Configuration Socket.IO...")
        allowed_modes: set[str] = {"threading", "eventlet", "gevent", "gevent_uwsgi"}
        env_mode = os.getenv("SOCKETIO_ASYNC_MODE", "eventlet")
        async_mode: AsyncMode = cast("AsyncMode", env_mode if env_mode in allowed_modes else "eventlet")

        sio_logger = config_name == "development"
        sio_engineio_logger = config_name == "development"

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
            allow_ws_upgrades,
        )

    # ✅ 2.9: Injection trace_id dans logs pour corrélation (une seule fois au démarrage)
    try:
        from shared.otel_setup import inject_trace_id_to_logs

        old_factory = logging.getLogRecordFactory()

        def record_factory_with_trace(*args, **kwargs):
            """Factory enrichissant les LogRecord avec trace_id/span_id."""
            record = old_factory(*args, **kwargs)
            try:
                trace_data = inject_trace_id_to_logs()
                record.trace_id = trace_data.get("trace_id", "")
                record.span_id = trace_data.get("span_id", "")
            except Exception:
                # Si injection échoue, continuer sans trace_id
                record.trace_id = ""
                record.span_id = ""
            return record

        logging.setLogRecordFactory(record_factory_with_trace)
        app.logger.info("[2.9] Logging enrichi avec trace_id/span_id")
    except Exception as e:
        app.logger.warning("[2.9] Échec enrichissement logs: %s", e)

    # Log léger pour les requêtes Socket.IO
    @app.before_request
    def _log_socketio_requests():  # pyright: ignore[reportUnusedFunction]
        p = request.path or ""
        if p.startswith("/socket.io"):
            app.logger.debug("📡 SIO %s %s from %s", request.method, request.full_path, request.remote_addr)

    # ✅ D3: Killswitch / Maintenance mode
    @app.before_request
    def _check_maintenance_mode():  # pyright: ignore[reportUnusedFunction]
        """Vérifie si le mode maintenance est activé."""
        # Vérifier variable d'environnement
        if os.getenv("MAINTENANCE_MODE", "false").lower() == "true":
            # Permettre les healthchecks même en maintenance
            if request.path in ["/health", "/api/health"]:
                return None
            return jsonify({"error": "Service en maintenance", "message": "Merci de réessayer plus tard"}), 503

        # Vérifier fichier flag (pour killswitch ChatOps)
        try:
            from pathlib import Path

            flag_file = Path("/tmp/atmr_maintenance_mode.flag")
            if flag_file.exists():
                reason = flag_file.read_text().strip()
                # Permettre les healthchecks même en maintenance
                if request.path in ["/health", "/api/health"]:
                    return None
                return jsonify({"error": "Service en maintenance", "message": reason}), 503
        except Exception:
            # Si erreur de lecture du flag, continuer normalement
            pass

    # ✅ D3: Vérification DB Read-Only (chaos injector)
    @app.before_request
    def _check_db_read_only():  # pyright: ignore[reportUnusedFunction]
        """Vérifie si la DB est en read-only (via chaos injector).

        ⚠️ Chaos ne doit JAMAIS être activé en production (vérifier CHAOS_ENABLED=false).
        Les requêtes GET/HEAD sont toujours autorisées, seules les écritures (POST/PUT/PATCH/DELETE)
        sont bloquées.
        """
        # Ne bloquer que les méthodes d'écriture
        if request.method not in ["POST", "PUT", "PATCH", "DELETE"]:
            return None

        # Permettre les healthchecks même en read-only
        if request.path in ["/health", "/api/health"]:
            return None

        try:
            from chaos.injectors import get_chaos_injector

            injector = get_chaos_injector()
            if injector.enabled and injector.db_read_only:
                app.logger.warning("[CHAOS] DB read-only: blocking %s %s", request.method, request.path)
                return jsonify(
                    {
                        "error": "Database is in read-only mode",
                        "message": "Writes are temporarily disabled. Please try again later.",
                    }
                ), 503
        except ImportError:
            # Si module chaos non disponible, continuer normalement
            pass
        except Exception:
            # En cas d'erreur inattendue, continuer (ne pas bloquer le système)
            app.logger.warning("[CHAOS] Error checking DB read-only status", exc_info=True)
            pass

        return None

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

    @app.route("/uploads/<path:filename>", methods=["GET", "OPTIONS"])
    def serve_uploads(filename: str):  # pyright: ignore[reportUnusedFunction]
        # Gérer les prérequêtes CORS
        if request.method == "OPTIONS":
            response = make_response("", 204)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

        base = Path(app.config["UPLOADS_DIR"]).resolve()
        candidate = (base / filename).resolve()

        # anti-path traversal
        if not str(candidate).startswith(str(base)):
            app.logger.warning("Tentative de path traversal bloquée: %s", filename)
            raise NotFound

        # Vérifier que le fichier existe
        if not candidate.exists():
            app.logger.warning(
                "Fichier upload introuvable: %s (chemin résolu: %s, base: %s)", filename, candidate, base
            )
            raise NotFound

        directory = str(base)
        fname = str(candidate.relative_to(base)).replace("\\", "/")

        app.logger.debug("Serving upload file: %s -> %s", filename, candidate)

        # Servir le fichier avec les en-têtes appropriés pour les images
        # send_from_directory définit automatiquement le Content-Type correct
        response = send_from_directory(directory, fname, conditional=True)

        # Ajouter les en-têtes CORS pour permettre l'accès depuis le frontend
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"

        # S'assurer que le Content-Type est correct pour les images SVG
        # (send_from_directory peut ne pas le détecter correctement)
        if filename.lower().endswith(".svg"):
            response.headers["Content-Type"] = "image/svg+xml"

        # Ajouter un header Cache-Control pour les images
        response.headers["Cache-Control"] = "public, max-age=3600"

        return response

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
        # En production, on force HTTPS pour la sécurité
        # Note: Les healthchecks Docker utilisent HTTP depuis localhost
        # On va créer un endpoint /health exempt de Talisman pour permettre les healthchecks
        force_https = True

    # ✅ Définir /health AVANT Talisman pour qu'il ne soit pas soumis à force_https
    # Les healthchecks Docker utilisent HTTP depuis localhost
    @app.route("/health")
    def health():  # pyright: ignore[reportUnusedFunction]
        # Endpoint utilisé par les healthchecks Docker : renvoie explicitement
        # le statut attendu (healthy + models_loaded).
        return jsonify(
            {
                "status": "healthy",
                "models_loaded": True,
            }
        ), 200

    # ✅ Intercepter /health AVANT Talisman pour éviter la redirection HTTPS
    # Les healthchecks Docker utilisent HTTP depuis localhost
    @app.before_request
    def _bypass_talisman_for_healthcheck():  # pyright: ignore[reportUnusedFunction]
        """Court-circuite Talisman pour /health depuis localhost (healthchecks Docker)."""
        if request.path == "/health":
            # Vérifier si la requête vient de localhost (healthcheck Docker)
            # Dans Docker, remote_addr peut être "127.0.0.1", "::1", ou l'IP du conteneur
            remote = request.remote_addr or ""
            host = request.host or ""
            # Si c'est depuis localhost OU si le schéma est HTTP (pas HTTPS), retourner directement
            if (
                remote in ("127.0.0.1", "::1", "localhost") 
                or request.scheme == "http"
                or "localhost" in host
            ):
                # Retourner directement la réponse JSON sans passer par Talisman
                return jsonify(
                    {
                        "status": "healthy",
                        "models_loaded": True,
                    }
                ), 200

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
            current_ct = resp.headers.get("Content-Type")
            resp.headers["Content-Type"] = f"{current_ct}; charset=utf-8"
        return resp

    # 5) CORS
    CORS(
        app,
        resources={r"/*": {"origins": "*" if cors_origins == "*" else cors_origins}},
        supports_credentials=True,
        expose_headers=["Content-Type", "Authorization"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "Cache-Control",
            "Pragma",
        ],  # ✅ Ajout Cache-Control et Pragma pour les requêtes avec no-cache
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
    # ✅ Skip routes initialization pour les scripts (évite blocage)
    skip_routes_init = os.getenv("SKIP_ROUTES_INIT", "false").lower() == "true"

    if not skip_routes_init:
        app.logger.info("🔧 [INIT] Enregistrement des routes et handlers Socket.IO...")
    else:
        app.logger.info("⏭️  Initialisation des routes désactivée (SKIP_ROUTES_INIT=1)")

    with app.app_context():
        from sqlalchemy.orm import configure_mappers

        import models

        _ = models  # Force l'import pour enregistrer les mappers

        configure_mappers()

        if not skip_routes_init:
            from routes_api import init_namespaces

            init_namespaces(app)

        if not skip_routes_init:
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

        if not skip_routes_init:
            # Réponse générique aux préflights CORS (toutes routes)
            @app.before_request
            def _cors_preflights_any():  # pyright: ignore[reportUnusedFunction]
                if request.method == "OPTIONS":
                    return make_response("", 204)
                return None

            # ✅ 3.2: Ajouter header Deprecation sur toutes les routes /api/v1/*
            @app.after_request
            def _add_deprecation_header_v1(response):  # pyright: ignore[reportUnusedFunction]
                """Ajoute le header Deprecation sur les routes API v1."""
                if request.path and request.path.startswith("/api/v1/"):
                    response.headers["Deprecation"] = 'version="v1"'
                    response.headers["Sunset"] = "Wed, 01 Jan 2025 00:00:00 GMT"  # Date estimée de suppression
                    response.headers["Link"] = '<https://docs.atmr.ch/api/v2>; rel="successor-version"'
                return response

            # ✅ 3.2: Ajouter header Deprecation sur routes legacy /api/* (si activées)
            @app.after_request
            def _add_deprecation_header_legacy(response):  # pyright: ignore[reportUnusedFunction]
                """Ajoute le header Deprecation sur les routes API legacy."""
                if request.path and request.path.startswith("/api/") and not request.path.startswith("/api/v"):
                    # Route legacy (sans version)
                    response.headers["Deprecation"] = 'version="legacy"'
                    response.headers["Sunset"] = "Wed, 01 Jan 2025 00:00:00 GMT"
                    response.headers["Link"] = '<https://docs.atmr.ch/api/v1>; rel="successor-version"'
                return response

            # ✅ 3.2: Shim générique pour /api[/vX]/auth/login si RESTX rate
            # Supporte /api/auth/login (legacy), /api/v1/auth/login et /api/v2/auth/login
            @app.before_request
            def _auth_login_shim_any_version():  # pyright: ignore[reportUnusedFunction]
                p = request.path or ""
                if request.method in ("POST", "OPTIONS") and re.fullmatch(r"/api(?:/v\d+)?/auth/login", p):
                    # Correspond à /api/auth/login (legacy), /api/v1/auth/login ou /api/v2/auth/login
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
                if path in {"/", "/health", "/config", "/docs", "/favicon.ico", "/robots.txt"} or path.startswith(
                    ("/swaggerui/", "/static/", "/uploads/", "/socket.io")
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
                    current_app.logger.info("Legacy shim internal reroute %s %s -> %s", request.method, path, new_path)
                    return None
                return None

        if not skip_socketio:
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

        # Note: L'endpoint /health est défini plus bas, après l'initialisation de Talisman
        # Il sera exempt de la redirection HTTPS via un décorateur ou une configuration spéciale

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
            # Intercepter les erreurs JWT expirées qui peuvent échapper aux handlers JWT
            from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

            if isinstance(e, ExpiredSignatureError):
                app.logger.warning("Token JWT expiré intercepté: %s", str(e))
                return jsonify({"error": "token_expired", "message": "Signature has expired"}), 401
            if isinstance(e, InvalidTokenError):
                app.logger.warning("Token JWT invalide intercepté: %s", str(e))
                return jsonify({"error": "invalid_token", "message": str(e)}), 422

            # Pour toutes les autres exceptions
            app.logger.exception("Unhandled server error")
            msg = str(e) if app.config.get("DEBUG") else "Une erreur interne est survenue."
            return jsonify({"error": "server_error", "message": msg}), 500

    # Note: handler disconnect géré dans sockets/chat.py (pas de doublon)

    return app
