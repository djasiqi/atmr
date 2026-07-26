# backend/app.py
# pyright: reportImportCycles = false

# ⚠️ CRITIQUE: Empêcher pytest de s'exécuter automatiquement en production
# Cette vérification doit être faite AVANT tout autre import
import os

if (
    os.getenv("FLASK_ENV", default=None) == "production"
    or os.getenv("FLASK_CONFIG", default=None) == "production"
):
    # Désactiver pytest en production
    os.environ["PYTEST_DISABLED"] = "1"
    os.environ["DISABLE_PYTEST"] = "1"
    # Empêcher pytest de collecter les tests
    os.environ["PYTEST_CURRENT_TEST"] = ""

import json
import logging
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

# TypeAlias import (compatible Python 3.10+)
try:
    from typing import TypeAlias

    # override n'est disponible dans typing qu'à partir de Python 3.12
    try:
        from typing import override
    except ImportError:
        from typing_extensions import override  # type: ignore[assignment]
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
import sentry_sdk  # noqa: F401 — réexport pour routes.* (from app import sentry_sdk)
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    current_app,
    jsonify,
    make_response,
    request,
)
from flask_cors import CORS
from flask_talisman import Talisman
from werkzeug.exceptions import (
    BadRequest,
    HTTPException,
    NotFound,
)
from werkzeug.middleware.proxy_fix import (
    ProxyFix,
)

# Extensions & config
from config import config
from ext import (
    _socketio_message_queue,
    bcrypt,
    db,
    jwt,
    limiter,
    mail,
    migrate,
    socketio,
)
from services.demo.environment_guard import (
    block_sensitive_integrations_in_demo,
    build_demo_environment_snapshot,
    enforce_demo_environment_or_raise,
)

# ---------- Chargement .env ----------
BASE_DIR = Path(__file__).resolve().parent
# Dépôt monorepo : souvent un `.env` à la racine (Docker, secrets partagés) et un
# `backend/.env` pour le service API. Charger la racine d'abord (sans écraser
# l'existant), puis `backend/.env` ; en dev ce dernier a la priorite sur les cles
# qu'il definit (ex. SAFERPAY_* uniquement dans la racine restent disponibles).
_repo_root_env = BASE_DIR.parent / ".env"
if _repo_root_env.is_file():
    load_dotenv(_repo_root_env, override=False)
if os.getenv("FLASK_ENV", default=None) == "development":
    load_dotenv(BASE_DIR / ".env", override=True)
else:
    # Production: ne jamais override les variables d'environnement système
    load_dotenv(BASE_DIR / ".env", override=False)

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
    # (on ne les stocke pas toutes dans required_vars, car on a déjà
    # la logique "au moins une des deux" juste en dessous)
    required_vars: set[str] = set()

    # Au moins une clé secrète est requise (JWT ou Flask)
    if not (
        os.getenv("JWT_SECRET_KEY", default=None)
        or os.getenv("SECRET_KEY", default=None)
    ):
        raise RuntimeError(
            "JWT_SECRET_KEY ou SECRET_KEY manquant(e). "
            + "Ajoutez au moins l'une de ces variables dans backend/.env "
            + "puis redémarrez."
        )

    # Variables critiques pour production
    if config_name == "production":
        # Variables **strictement** requises en prod (hors Redis/DB)
        production_vars: set[str] = set()

        required_vars.update(production_vars)

        # Vérifier que la configuration de base de données est disponible
        has_db_config = (
            os.getenv("DATABASE_URL", default=None)
            or os.getenv("SQLALCHEMY_DATABASE_URI", default=None)
            or (
                os.getenv("POSTGRES_USER", default=None)
                and os.getenv("POSTGRES_PASSWORD", default=None)
                and os.getenv("POSTGRES_DB", default=None)
            )
        )
        if not has_db_config:
            raise RuntimeError(
                "Configuration de base de données manquante pour production. "
                + "Fournissez soit DATABASE_URL, soit SQLALCHEMY_DATABASE_URI, "
                + "soit POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB."
            )

        # ✅ Redis : soit REDIS_URL, soit au moins REDIS_PASSWORD
        has_redis_config = os.getenv("REDIS_URL", default=None) or os.getenv(
            "REDIS_PASSWORD", default=None
        )
        if not has_redis_config:
            raise RuntimeError(
                "Configuration Redis manquante pour production. "
                + "Fournissez soit REDIS_URL, soit REDIS_PASSWORD."
            )

        # ✅ S1: Valider CORS en production - rejeter configuration "*"
        cors_origins_env = os.getenv("SOCKETIO_CORS_ORIGINS", default="")
        if not cors_origins_env or cors_origins_env.strip() == "*":
            raise RuntimeError(
                "Configuration CORS invalide pour production. "
                + "SOCKETIO_CORS_ORIGINS ne peut pas être vide ou '*' "
                + "en production. Fournissez une liste d'origines autorisées "
                + "séparées par des virgules (ex: 'https://app.example.com,"
                + "https://www.example.com')."
            )
        # Valider que toutes les origines sont HTTPS (sauf localhost pour tests)
        origins_list = [origin.strip() for origin in cors_origins_env.split(",")]
        for origin in origins_list:
            if origin and not origin.startswith(
                ("https://", "http://localhost", "http://127.0.0.1")
            ):
                raise RuntimeError(
                    "SOCKETIO_CORS_ORIGINS contient une origine non sécurisée "
                    + f"en production: {origin}. Toutes les origines doivent "
                    + "être HTTPS (sauf localhost pour tests)."
                )

        # ✅ Valider PDF_BASE_URL en production (REQUIS)
        # Exception: localhost/127.0.0.1 autorisé en développement local
        pdf_base_url = os.getenv("PDF_BASE_URL", default=None)
        if not pdf_base_url:
            raise RuntimeError(
                "PDF_BASE_URL est requis en production. "
                + "Définissez cette variable d'environnement avec une URL HTTPS valide."
            )
        # ✅ Exception pour développement local (localhost/127.0.0.1)
        is_localhost = pdf_base_url.startswith(
            "http://localhost"
        ) or pdf_base_url.startswith("http://127.0.0.1")
        if not pdf_base_url.startswith("https://") and not is_localhost:
            raise RuntimeError(
                "PDF_BASE_URL doit commencer par 'https://' en production. "
                + f"Valeur actuelle: {pdf_base_url}"
            )

        # Admin Ops : GET /api/v1/platform/status — whitelist IP obligatoire si demandé
        wl_required = os.getenv(
            "ADMIN_IP_WHITELIST_REQUIRED", "false"
        ).strip().lower() in (
            "true",
            "1",
            "yes",
        )
        wl_raw = (os.getenv("ADMIN_IP_WHITELIST") or "").strip()
        if wl_required and not wl_raw:
            import logging

            _wl_log = logging.getLogger(__name__)
            _wl_log.error(
                "Sécurité : ADMIN_IP_WHITELIST_REQUIRED=true mais "
                + "ADMIN_IP_WHITELIST est vide. Définissez au moins une IP ou un "
                + "CIDR autorisé pour GET /api/v1/platform/status (console Admin Ops)."
            )
            raise RuntimeError(
                "Sécurité : en production, ADMIN_IP_WHITELIST_REQUIRED=true exige "
                + "ADMIN_IP_WHITELIST non vide (accès /api/v1/platform/status)."
            )

        # SENTRY_DSN est optionnel mais recommandé
        recommended_vars = {
            "SENTRY_DSN",
        }

        # Vérifier les variables requises
        missing: list[str] = []
        for var in required_vars:
            if not os.getenv(var, default=None):
                missing.append(var)

        if missing:
            missing_str = ", ".join(missing)
            error_msg = (
                f"Variables d'environnement manquantes pour production: "
                f"{missing_str}\n"
                f"Ajoutez-les dans backend/.env puis redémarrez."
            )
            raise RuntimeError(error_msg)

        # Vérifier variables recommandées et avertir si manquantes
        missing_recommended = [
            var for var in recommended_vars if not os.getenv(var, default=None)
        ]
        if missing_recommended:
            import logging

            logger = logging.getLogger(__name__)
            missing_str = ", ".join(missing_recommended)
            warning_msg = (
                "Variables recommandées manquantes pour production: %s. "
                "L'application fonctionnera mais certaines fonctionnalités "
                "peuvent être limitées."
            )
            logger.warning(warning_msg, missing_str)

        # F-01 : secret ingest GPS + TTL idempotence (module sans cycle routes)
        from services.security.internal_service_auth import (
            validate_internal_service_token_for_boot,
        )

        validate_internal_service_token_for_boot(config_name=config_name)

        # F-03 : fenêtre legacy activation (FROM/UNTIL absolus, sans cycle routes)
        from services.security.activation_legacy import (
            validate_activation_legacy_for_boot,
        )

        validate_activation_legacy_for_boot(config_name=config_name)

    # Variables critiques pour la stack démo
    if config_name == "demo" or (os.getenv("APP_ENV", "").strip().lower() == "demo"):
        if os.getenv("APP_ENV", "").strip().lower() != "demo":
            raise RuntimeError("FLASK_CONFIG=demo exige APP_ENV=demo.")
        if os.getenv("DEMO_MODE", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError("FLASK_CONFIG=demo exige DEMO_MODE=true.")


# ---------- JSON encoder/provider ----------
class CustomJSONEncoder(json.JSONEncoder):
    @override
    def default(self, o):  # pyright: ignore[reportImplicitOverride]
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


# Cache in-memory (par processus worker) pour le fichier flag maintenance : évite stat()/read
# sur chaque requête. TTL 5s — aligné sur la propagation type ChatOps.
_MAINTENANCE_FLAG_CACHE_TTL_SEC = 5.0
_maintenance_flag_file_cache: dict[str, Any] = {
    "at": 0.0,
    "file_active": False,
    "reason": "",
}


def _check_maintenance_flag_cached() -> tuple[bool, str]:
    """Lecture du flag fichier maintenance avec cache TTL 5s (time.monotonic()).

    Returns:
        (file_active, reason): actif si le fichier impose un 503, message pour le body JSON.

    En cas d'erreur FS/lecture: pas de 503 fichier, sans cacher l'échec (re-tentative au prochain appel).
    """
    c = _maintenance_flag_file_cache
    now = time.monotonic()
    if c["at"] and (now - c["at"] < _MAINTENANCE_FLAG_CACHE_TTL_SEC):
        return c["file_active"], c["reason"]

    try:
        maintenance_tmp_dir = os.getenv("MAINTENANCE_TMP_DIR", "/tmp")
        flag_file = Path(maintenance_tmp_dir) / "atmr_maintenance_mode.flag"
        if not flag_file.exists():
            c["at"] = now
            c["file_active"] = False
            c["reason"] = ""
            return False, ""
        c["at"] = now
        c["file_active"] = True
        c["reason"] = flag_file.read_text().strip()
        return c["file_active"], c["reason"]
    except Exception:
        c["at"] = 0.0
        return False, ""


def _is_maintenance_sane_bypass_path() -> bool:
    """Retourne True pour court-circuiter le flag fichier maintenance (I/O / 503).

    - Health : inchangé vis-à-vis de l'exemption historique.
    - OPTIONS sur statique / bruit : évite d'ouvrir le flag pour des pré-vols.
    L'environnement ``MAINTENANCE_MODE=true`` est géré **avant** ce court-circuit.
    """
    from flask import request

    p = request.path or ""
    return p in ("/health", "/api/health", "/api/v1/health") or (
        request.method == "OPTIONS"
        and (
            p.startswith(("/static/", "/swaggerui/", "/uploads/"))
            or p in ("/favicon.ico", "/robots.txt")
        )
    )


# Cache in-process (par worker) : public_id JWT -> user_id pour _track_unauthorized_access.
# Évite un User.query à chaque 401/403 lorsque le même jeton génère des refus répétés.
_UNAUTHORIZED_ACCESS_USER_ID_CACHE: dict[str, tuple[int | None, float]] = {}
_UNAUTHORIZED_ACCESS_USER_ID_TTL_SEC = 60.0
_UNAUTHORIZED_ACCESS_USER_ID_MAX = 256


def _prune_unauthorized_access_user_id_cache(now: float) -> None:
    c = _UNAUTHORIZED_ACCESS_USER_ID_CACHE
    expired = [
        k
        for k, (_, ts) in c.items()
        if now - ts >= _UNAUTHORIZED_ACCESS_USER_ID_TTL_SEC
    ]
    for k in expired:
        del c[k]
    while len(c) > _UNAUTHORIZED_ACCESS_USER_ID_MAX // 2:
        c.pop(next(iter(c)), None)


def _resolve_user_id_for_unauthorized_access_alert() -> int | None:
    """Résout l'ID interne utilisateur pour les alertes 401/403, sans requête DB inutile.

    - Pas d'identité JWT exploitable : None (aucun accès à User).
    - Identité présente : requête DB ou entrée de cache court (même public_id, fenêtre TTL).
    """
    try:
        from flask_jwt_extended import get_jwt_identity

        public_id = get_jwt_identity()
    except Exception:
        return None
    if not public_id:
        return None
    now = time.monotonic()
    if public_id in _UNAUTHORIZED_ACCESS_USER_ID_CACHE:
        uid, ts = _UNAUTHORIZED_ACCESS_USER_ID_CACHE[public_id]
        if now - ts < _UNAUTHORIZED_ACCESS_USER_ID_TTL_SEC:
            return uid
    try:
        from models import User

        user = User.query.filter_by(public_id=public_id).first()
        uid = user.id if user else None
    except Exception:
        return None
    if len(_UNAUTHORIZED_ACCESS_USER_ID_CACHE) >= _UNAUTHORIZED_ACCESS_USER_ID_MAX:
        _prune_unauthorized_access_user_id_cache(now)
    _UNAUTHORIZED_ACCESS_USER_ID_CACHE[public_id] = (uid, now)
    return uid


def create_app(config_name: str | None = None):
    if config_name is None:
        config_name = os.getenv("FLASK_CONFIG", "development")

    # Valider variables d'environnement critiques avant de continuer
    validate_required_env_vars(config_name)

    app = Flask(__name__)
    from middleware.silent_json_request import (
        SilentJSONRequest,
        register_json_body_precache,
    )

    app.request_class = SilentJSONRequest
    # F-01 : garde body bornée AVANT le précache JSON global
    from middleware.internal_tracking_body_guard import (
        register_internal_tracking_ingest_body_guard,
    )

    register_internal_tracking_ingest_body_guard(app)
    register_json_body_precache(app)

    # Désactiver les slashes stricts pour éviter les redirections 308
    app.url_map.strict_slashes = False
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    # Utiliser config_name plutôt que FLASK_ENV: en tests on peut créer une app
    # "production" avec FLASK_ENV=testing.
    if config_name != "production":
        app.config["PREFERRED_URL_SCHEME"] = "http"
        app.config["SESSION_COOKIE_SECURE"] = False
        # ✅ En développement, permettre les cookies sur http://localhost
        app.config["COOKIE_SECURE"] = False
        app.config["JWT_COOKIE_SECURE"] = False
    else:
        # Sécurité cookies de session en production
        app.config["SESSION_COOKIE_SECURE"] = True  # HTTPS uniquement
        app.config["SESSION_COOKIE_HTTPONLY"] = True
        app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Flask 2.3+ : JSON Provider ; fallback <2.3 : json_encoder
    try:
        from flask.json.provider import (
            DefaultJSONProvider,
        )

        class CustomJSONProvider(DefaultJSONProvider):
            @override
            def default(self, o):
                if isinstance(o, Enum):
                    return o.value
                return super().default(o)

        app.json = CustomJSONProvider(app)
    except Exception:
        # Fallback pour Flask < 2.3 (json_encoder est déprécié mais fonctionne)
        app.json_encoder = CustomJSONEncoder

    # 1) Config
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Guard fail-fast demo/prod au runtime
    snapshot = build_demo_environment_snapshot()
    enforce_demo_environment_or_raise(snapshot)
    if snapshot.app_env == "demo":
        for feature, blocked in block_sensitive_integrations_in_demo().items():
            app.config[f"DEMO_BLOCK_{feature.upper()}"] = blocked

    # ✅ Validation de sécurité en production
    from config import validate_production_security

    validate_production_security(app)

    from services.saferpay.config import warn_saferpay_test_api_url_in_production

    warn_saferpay_test_api_url_in_production(app.logger, config_name=config_name)

    # ✅ Force UTF-8 encoding pour JSON et réponses
    app.config["JSON_AS_ASCII"] = False
    app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"

    # 2) Extensions
    db.init_app(app)
    jwt.init_app(app)
    mail.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db, compare_type=True, render_as_batch=True)

    # ✅ A2: Détection N+1 queries en développement
    if app.config.get("ENV") == "development" or app.config.get("SQLALCHEMY_ECHO"):
        try:
            from nplusone.ext.flask_sqlalchemy import (
                NPlusOne,
            )

            NPlusOne(app)
            app.logger.info("✅ NPlusOne activé - Détection N+1 queries activée")
        except ImportError:
            app.logger.warning(
                "⚠️ nplusone non installé - Exécuter: pip install nplusone"
            )

    # ✅ Clean Architecture: wiring explicite du bus d'événements
    # - Application ne dépend pas de services/celery/redis.
    # - Le choix de l'implémentation (infra) est fait ici (outer layer).
    try:
        from application.events.event_bus import set_event_bus
        from infrastructure.events.celery_event_bus import CeleryEventBus
        from infrastructure.events.in_memory_event_bus import InMemoryEventBus

        default_mode = "celery" if config_name == "production" else "in_memory"
        mode = os.getenv("EVENT_BUS_MODE", default_mode).strip().lower()
        if mode in ("celery", "async"):
            set_event_bus(CeleryEventBus())
            app.logger.info("[EventBus] ✅ CeleryEventBus activé (mode=%s)", mode)
        else:
            set_event_bus(InMemoryEventBus())
            app.logger.info("[EventBus] ✅ InMemoryEventBus activé (mode=%s)", mode)
    except Exception as e:
        # Ne pas bloquer le démarrage si le wiring échoue
        # (tests/unit, dépendances optionnelles, etc.)
        app.logger.warning("[EventBus] ⚠️ Wiring bus échoué: %s", e)

    if app.config.get("RATELIMIT_ENABLED", True):
        limiter.init_app(app)

    # ✅ S1: Setup protection CSRF pour requêtes mutantes
    try:
        from services.security.csrf import setup_csrf_protection

        # Prod Lot 1-D : CSRF forcé
        env_name = (os.getenv("ENVIRONMENT") or config_name or "").strip().lower()
        if env_name == "production":
            os.environ["CSRF_ENABLED"] = "true"
            app.config["CSRF_ENABLED"] = True
            csrf_enabled = True
        else:
            csrf_enabled = os.getenv("CSRF_ENABLED", "true").lower() in (
                "true",
                "1",
                "yes",
            )
        if csrf_enabled:
            setup_csrf_protection(app)
            app.logger.info("[CSRF] ✅ Protection CSRF activée")
        else:
            app.logger.info("[CSRF] ⚠️ Protection CSRF désactivée (CSRF_ENABLED=false)")
    except Exception as e:
        app.logger.warning("[CSRF] ⚠️ Échec activation protection CSRF: %s", e)

    # Lot 1-E : web = cookies ; mobile / endpoints dédiés = Bearer headers only
    # (cookies web présents ignorés pour l'auth)
    @app.before_request
    def _lot1_jwt_locations_web_mobile_split():  # pyright: ignore[reportUnusedFunction]
        path = request.path or ""
        mobile_path = path.startswith(
            (
                "/api/v1/company_mobile/",
                "/api/v1/driver/",
            )
        )
        is_expo = request.headers.get("X-Requested-With") == "Expo"
        ua = (request.headers.get("User-Agent") or "").lower()
        mobile_ua = any(
            m in ua for m in ("expo", "okhttp", "darwin", "cfnetwork", "dalvik")
        )
        if mobile_path or is_expo or mobile_ua:
            app.config["JWT_TOKEN_LOCATION"] = ["headers"]
        else:
            app.config["JWT_TOKEN_LOCATION"] = ["cookies", "headers"]

    # ✅ 2.9: Setup OpenTelemetry avec instrumentation complète
    try:
        from shared.otel_setup import (
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
        # L'engine peut ne pas être disponible immédiatement (lazy creation)
        try:
            engine = None
            # Essayer d'obtenir l'engine dans un contexte d'application
            with app.app_context():
                try:
                    _eng = getattr(db, "engine", None)
                    if _eng is not None:
                        # Flask-SQLAlchemy 3+ utilise .engine directement
                        engine = _eng
                    elif hasattr(db, "get_engine"):
                        # Flask-SQLAlchemy <3 utilise get_engine() avec app context
                        engine = db.get_engine()
                except Exception:
                    # Engine pas encore créé, on utilisera l'instrumentation différée
                    engine = None

            if engine:
                # Instrumenter immédiatement si l'engine est disponible
                with app.app_context():
                    instrument_sqlalchemy(engine)
            else:
                # Délayer l'instrumentation après que l'engine soit créé
                # Utiliser before_request avec un flag au lieu de before_first_request
                # (déprécié dans Flask 2.2+)
                _db_instrumented = {"done": False}

                @app.before_request
                def _instrument_db_after_init():  # pyright: ignore
                    if _db_instrumented["done"]:
                        return
                    try:
                        with app.app_context():
                            _db_eng = getattr(db, "engine", None)
                            if _db_eng is not None:
                                db_engine = _db_eng
                            elif hasattr(db, "get_engine"):
                                db_engine = db.get_engine()
                            else:
                                return
                            instrument_sqlalchemy(db_engine)
                            _db_instrumented["done"] = True
                    except Exception as e:
                        app.logger.warning(
                            "[2.9] Échec instrumentation DB différée: %s", e
                        )
        except Exception as e:
            # Ne pas logger comme warning si c'est juste que l'engine
            # n'est pas encore créé
            if "application context" not in str(e).lower():
                app.logger.warning("[2.9] Échec instrumentation SQLAlchemy: %s", e)
    except ImportError as e:
        # Ne pas logger le warning en mode test (évite le bruit dans les tests)
        if (
            os.getenv("FLASK_ENV", "") != "testing"
            and os.getenv("FLASK_CONFIG", "") != "testing"
        ):
            app.logger.warning("[2.9] OpenTelemetry non disponible (optionnel): %s", e)
    except Exception as e:
        app.logger.warning("[2.9] Échec initialisation OpenTelemetry: %s", e)

    # ✅ 2.7: Setup DB Profiler pour détecter N+1
    # (activable via ENABLE_DB_PROFILING=true)
    try:
        from ext import setup_db_profiler

        setup_db_profiler(app)
    except Exception as e:
        app.logger.warning("[DB Profiler] Échec initialisation: %s", e)

    # ✅ Phase 3: Setup centralisation des logs (ELK/Loki)
    try:
        from shared.logging_centralized import setup_centralized_logging

        setup_centralized_logging(app)
    except Exception as e:
        app.logger.warning("[Centralized Logging] Échec initialisation: %s", e)

    # ✅ P0 Stabilization: Setup trace_id middleware
    try:
        from middleware.trace_id import (
            add_trace_id_to_response,
            inject_trace_id_middleware,
        )

        @app.before_request
        def inject_trace_id():  # pyright: ignore
            """Injecte trace_id avant chaque requête."""
            inject_trace_id_middleware()

        @app.after_request
        def add_trace_id_header(response):  # pyright: ignore
            """Ajoute trace_id dans les headers de réponse."""
            return add_trace_id_to_response(response)

        app.logger.info("[P0] Trace ID middleware activé")
    except Exception as e:
        app.logger.warning("[P0] Échec initialisation trace_id middleware: %s", e)

    try:
        from services.monitoring.company_dashboard_sql_perf import (
            init_company_dashboard_sql_perf,
        )

        init_company_dashboard_sql_perf(app)
    except Exception as e:
        app.logger.warning("[CompanyDashboardSQLPerf] Échec initialisation: %s", e)

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
    from services.infrastructure.runtime_flags import is_skip_socketio

    skip_socketio = is_skip_socketio()

    # Définir cors_origins même si Socket.IO est désactivé
    # (nécessaire pour CORS plus bas)
    # ✅ Liste centralisée des origines CORS en développement (pas de "*" + headers custom).
    # Dev local : 3000 = dashboard company, 8081 = app chauffeur web (Bearer only, withCredentials=false).
    # Pour éviter ERR_NETWORK sur le driver web (8081), le backend doit tourner avec FLASK_CONFIG=development
    # ou SOCKETIO_CORS_ORIGINS contenant http://localhost:8081 et http://127.0.0.1:8081.
    CORS_DEV_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8081",  # App chauffeur web (Expo)
        "http://127.0.0.1:8081",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://localhost:19000",
        "http://127.0.0.1:19000",
    ]

    if config_name == "development":
        # ✅ En développement, spécifier explicitement les origines locales
        # (ne peut pas utiliser "*" avec cors_credentials=True)
        # Permettre override via variable d'environnement,
        # mais toujours inclure les origines essentielles
        dev_cors_origins_env = os.getenv("SOCKETIO_CORS_ORIGINS", default="")
        if dev_cors_origins_env:
            # Parser les origines depuis l'environnement
            env_origins = [
                origin.strip()
                for origin in dev_cors_origins_env.split(",")
                if origin.strip()
            ]
            # Fusionner avec les origines essentielles (éviter les doublons)
            cors_origins = list(set(CORS_DEV_ORIGINS + env_origins))
        else:
            # Utiliser la liste centralisée par défaut
            cors_origins = CORS_DEV_ORIGINS.copy()
        # ✅ Logger les origines autorisées pour debug
        app.logger.info(
            "[Socket.IO] CORS configuré en développement avec %d origine(s): %s",
            len(cors_origins),
            ", ".join(cors_origins),
        )
    else:
        cors_origins_str = os.getenv("SOCKETIO_CORS_ORIGINS", default="")
        if cors_origins_str:
            cors_origins = []
            for origin_raw in cors_origins_str.split(","):
                origin_normalized = origin_raw.strip()
                if origin_normalized:
                    # ✅ Normaliser les origines HTTPS : supprimer :443 si présent
                    # Car HTTPS n'affiche normalement pas le port
                    if (
                        origin_normalized.startswith("https://")
                        and ":443" in origin_normalized
                    ):
                        origin_normalized = origin_normalized.replace(":443", "")
                    cors_origins.append(origin_normalized)
                    # ✅ Ajouter aussi la variante avec :443 pour accepter les deux formats
                    # Certains clients envoient explicitement le port 443
                    if (
                        origin_normalized.startswith("https://")
                        and ":443" not in origin_normalized
                    ):
                        cors_origins.append(origin_normalized + ":443")
            # ✅ Logger les origines autorisées pour debug
            app.logger.info(
                "[Socket.IO] CORS configuré avec %d origine(s): %s",
                len(cors_origins),
                ", ".join(cors_origins),
            )
        else:
            cors_origins = []

    # En dev/test uniquement : autoriser tous les headers CORS pour débloquer vite en local.
    # En prod, on garde la liste explicite (sécurité).
    cors_allow_headers_wildcard = config_name in ("development", "testing")

    if skip_socketio:
        app.logger.info("⏭️  Socket.IO désactivé (SKIP_SOCKETIO=1)")
    else:
        app.logger.info("🔧 [INIT] Configuration Socket.IO...")
        allowed_modes: set[str] = {"threading", "eventlet", "gevent", "gevent_uwsgi"}

        # ✅ FIX: Si DISABLE_EVENTLET=1, forcer threading mode pour éviter
        # que Flask-SocketIO n'initialise eventlet et casse les migrations Alembic
        _disable_eventlet = os.getenv("DISABLE_EVENTLET", "0") == "1"
        if _disable_eventlet:
            env_mode = "threading"
            app.logger.info("[Socket.IO] Mode threading forcé (DISABLE_EVENTLET=1)")
        else:
            env_mode = os.getenv("SOCKETIO_ASYNC_MODE", "gevent")
        async_mode: AsyncMode = cast(
            "AsyncMode", env_mode if env_mode in allowed_modes else "gevent"
        )

        sio_logger = config_name == "development"
        sio_engineio_logger = config_name == "development"

        # Par défaut : autoriser le WebSocket (aligné avec docker-compose : SIO_DISABLE_UPGRADES=false).
        # Sinon les clients RN qui tentent le WS en premier reçoivent « websocket error » alors que REST marche.
        # Pour désactiver en dev ponctuel : SIO_DISABLE_UPGRADES=true
        allow_ws_upgrades = True
        if (
            config_name == "development"
            and os.getenv("SIO_DISABLE_UPGRADES", "false").lower() == "true"
        ):
            allow_ws_upgrades = False

        ping_timeout_seconds = int(os.getenv("SOCKETIO_PING_TIMEOUT_SECONDS", "120"))
        ping_interval_seconds = int(os.getenv("SOCKETIO_PING_INTERVAL_SECONDS", "25"))

        # NB: pas de 'upgrade_timeout' (paramètre inexistant)
        # ni 'cookie=True' (type incompatible)
        # ✅ FIX Socket.IO multi-workers: Passer message_queue explicitement
        # à init_app() pour garantir que Redis est utilisé pour partager
        # les SIDs entre workers.
        # Le message_queue est configuré dans ext.py, mais doit être passé ici aussi.
        test_disable_message_queue = bool(
            app.config.get("SOCKETIO_TEST_DISABLE_MESSAGE_QUEUE", False)
        )
        configured_message_queue = (
            None
            if app.config.get("TESTING") and test_disable_message_queue
            else _socketio_message_queue
        )
        socketio.init_app(
            app,
            async_mode=async_mode,
            cors_allowed_origins=cors_origins if cors_origins else [],
            path="/socket.io",
            logger=sio_logger,
            engineio_logger=sio_engineio_logger,
            ping_timeout=ping_timeout_seconds,
            ping_interval=ping_interval_seconds,
            max_http_buffer_size=10_000_000,  # int
            allow_upgrades=allow_ws_upgrades,
            cors_credentials=True,
            # ✅ FIX: Ré-activer Redis pour synchroniser les handlers entre workers Gunicorn
            message_queue=configured_message_queue,
            # ✅ Note: La compression Socket.IO est gérée automatiquement
            # par le protocole lors de la négociation client/serveur.
            # Pas besoin de paramètre explicite dans Flask-SocketIO.
        )

        msg = (
            "✅ Socket.IO initialisé: async_mode=%s, cors=%s, "
            "allow_upgrades=%s, path=/socket.io"
        )
        app.logger.info(
            msg,
            async_mode,
            "*"
            if (isinstance(cors_origins, str) and cors_origins == "*")
            else "restricted",
            allow_ws_upgrades,
        )
        # ✅ Log explicite pour confirmer que Socket.IO est branché sur l'app
        print(
            "🔌 [SOCKET.IO] Flask-SocketIO initialisé avec path=/socket.io", flush=True
        )
        try:
            from services.infrastructure.socketio_runtime_check import (
                log_socketio_runtime_diagnostics,
            )

            log_socketio_runtime_diagnostics(
                app.logger,
                message_queue=configured_message_queue,
            )
        except Exception as exc:
            app.logger.warning(
                "[Socket.IO] Diagnostics runtime non disponibles: %s", exc
            )

    # ✅ 2.9: Injection trace_id dans logs pour corrélation
    # (une seule fois au démarrage)
    try:
        import threading

        from shared.otel_setup import inject_trace_id_to_logs

        old_factory = logging.getLogRecordFactory()

        # ✅ Protection contre récursion infinie
        _factory_lock = threading.local()

        def record_factory_with_trace(*args, **kwargs):
            """Factory enrichissant les LogRecord avec trace_id/span_id."""
            # Protection contre récursion : si on est déjà dans la factory, utiliser l'ancienne
            if hasattr(_factory_lock, "in_factory"):
                return old_factory(*args, **kwargs)

            _factory_lock.in_factory = True
            try:
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
            finally:
                _factory_lock.in_factory = False

        logging.setLogRecordFactory(record_factory_with_trace)
        app.logger.info("[2.9] Logging enrichi avec trace_id/span_id")
    except Exception as e:
        app.logger.warning("[2.9] Échec enrichissement logs: %s", e)

    # Log léger pour les requêtes Socket.IO
    @app.before_request
    def _log_socketio_requests():  # pyright: ignore[reportUnusedFunction]
        p = request.path or ""
        if p.startswith("/socket.io"):
            app.logger.debug(
                "📡 SIO %s %s from %s",
                request.method,
                request.full_path,
                request.remote_addr,
            )

    # ✅ D3: Killswitch / Maintenance mode
    @app.before_request
    def _check_maintenance_mode():  # pyright: ignore[reportUnusedFunction]
        """Vérifie si le mode maintenance est activé."""
        # Vérifier variable d'environnement
        if os.getenv("MAINTENANCE_MODE", "false").lower() == "true":
            # Permettre les healthchecks même en maintenance
            if request.path in ["/health", "/api/health", "/api/v1/health"]:
                return None
            return jsonify(
                {
                    "error": "Service en maintenance",
                    "message": "Merci de réessayer plus tard",
                }
            ), 503

        # Même exemption que l'historique fichier, sans lire le flag (early-exit)
        if _is_maintenance_sane_bypass_path():
            return None

        # Fichier flag (killswitch ChatOps) — cache TTL 5s, pas de stat() à chaque requête
        file_active, file_reason = _check_maintenance_flag_cached()
        if file_active:
            if request.path in ["/health", "/api/health", "/api/v1/health"]:
                return None
            return jsonify(
                {"error": "Service en maintenance", "message": file_reason}
            ), 503

        return None

    # ✅ S3: Middleware pour capturer les tentatives d'accès non autorisé
    @app.after_request
    def _track_unauthorized_access(resp):  # pyright: ignore[reportUnusedFunction]
        """✅ S3: Enregistre les tentatives d'accès non autorisé (401/403)
        pour alertes."""
        if resp.status_code in (401, 403):
            try:
                from security.security_alerts import SecurityAlertService

                ip_address = request.remote_addr or "unknown"
                endpoint = request.path or "unknown"
                user_id = _resolve_user_id_for_unauthorized_access_alert()

                # Enregistrer la tentative pour détection d'alertes
                SecurityAlertService.record_unauthorized_access(
                    ip_address=ip_address,
                    endpoint=endpoint,
                    status_code=resp.status_code,
                    user_id=user_id,
                )
            except Exception as e:
                # Ne pas bloquer la réponse si le tracking échoue
                app.logger.debug(
                    "[SecurityAlerts] Failed to track unauthorized access: %s", e
                )

        return resp

    # ✅ D3: Vérification DB Read-Only (chaos injector)
    @app.before_request
    def _check_db_read_only():  # pyright: ignore[reportUnusedFunction]
        """Vérifie si la DB est en read-only (via chaos injector).

        ⚠️ Chaos ne doit JAMAIS être activé en production
        (vérifier CHAOS_ENABLED=false).
        Les requêtes GET/HEAD sont toujours autorisées,
        seules les écritures (POST/PUT/PATCH/DELETE) sont bloquées.
        """
        # Ne bloquer que les méthodes d'écriture
        if request.method not in ["POST", "PUT", "PATCH", "DELETE"]:
            return None

        # Permettre les healthchecks même en read-only
        if request.path in ["/health", "/api/health", "/api/v1/health"]:
            return None

        try:
            from chaos.injectors import get_chaos_injector

            injector = get_chaos_injector()
            if injector.enabled and injector.db_read_only:
                app.logger.warning(
                    "[CHAOS] DB read-only: blocking %s %s",
                    request.method,
                    request.path,
                )
                return jsonify(
                    {
                        "error": "Database is in read-only mode",
                        "message": (
                            "Writes are temporarily disabled. Please try again later."
                        ),
                    }
                ), 503
        except ImportError:
            # Si module chaos non disponible, continuer normalement
            pass
        except Exception:
            # En cas d'erreur inattendue, continuer (ne pas bloquer le système)
            app.logger.warning(
                "[CHAOS] Error checking DB read-only status", exc_info=True
            )

        return None

    # à l'initialisation Flask
    @app.teardown_appcontext
    def shutdown_session(  # pyright: ignore[reportUnusedFunction]
        exception=None,  # noqa: ARG001
    ):
        # ✅ FIX: Gérer les connexions PostgreSQL fermées côté serveur
        # (stale connections dans les workers Celery)
        import contextlib

        try:
            db.session.remove()
        except Exception as e:
            # Si la connexion est déjà fermée (OperationalError),
            # on log et on continue sans bloquer
            app.logger.warning(
                "[shutdown_session] Error removing session (stale connection?): %s",
                str(e)[:200],
            )
            # Forcer la fermeture de la connexion sans rollback
            with contextlib.suppress(Exception):
                db.session.rollback()
            with contextlib.suppress(Exception):
                db.session.close()

    # Gestion d'erreurs réseau (Bad file descriptor, etc.)
    # Constantes pour les codes d'erreur système
    BAD_FILE_DESCRIPTOR = 9

    @app.errorhandler(OSError)
    def handle_os_error(e: OSError):  # pyright: ignore[reportUnusedFunction]
        err = getattr(e, "errno", None)
        if err == BAD_FILE_DESCRIPTOR:  # Bad file descriptor
            app.logger.warning(
                "Socket error (Bad file descriptor) - connection may have been closed"
            )
            return jsonify({"error": "Connection closed"}), 499
        raise e

    # 3) Uploads — chemin absolu pour dev local + Docker (évite ERR_CONNECTION_RESET)
    uploads_root = Path(app.root_path, "uploads").resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)
    (uploads_root / "invoices").mkdir(parents=True, exist_ok=True)
    app.config.setdefault("UPLOADS_DIR", str(uploads_root))
    app.config.setdefault("UPLOAD_FOLDER", str(uploads_root))
    app.config.setdefault("UPLOADS_PUBLIC_BASE", "/uploads")

    @app.route("/uploads/<path:filename>", methods=["GET", "OPTIONS"])
    def serve_uploads(filename: str):  # pyright: ignore[reportUnusedFunction]
        # Gérer les prérequêtes CORS
        if request.method == "OPTIONS":
            response = make_response("", 204)
            response.headers["X-Content-Type-Options"] = "nosniff"
            origin = request.headers.get("Origin")
            if origin and origin in cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

        # Lot 0 SEC-06: seuls les logos restent publics sous /uploads
        from shared.upload_path_resolver import (
            is_public_upload_prefix,
            resolve_safe_upload_path,
        )

        if not is_public_upload_prefix(filename):
            raise NotFound

        base = Path(app.config["UPLOADS_DIR"]).resolve()
        try:
            candidate = resolve_safe_upload_path(filename, uploads_base=base)
        except NotFound:
            app.logger.warning(
                "Upload public refusé ou introuvable: filename=%s base=%s",
                filename,
                base,
            )
            raise

        app.logger.info("Serving public upload file: %s -> %s", filename, candidate)

        # Contournement Gunicorn+eventlet : send_from_directory provoque ERR_CONNECTION_RESET
        # sur les fichiers (streaming incompatible). Lecture en mémoire + Response évite le bug.
        mimetypes_map = {
            ".pdf": "application/pdf",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        ext = Path(filename).suffix.lower()
        mimetype = mimetypes_map.get(ext)
        if mimetype is None:
            import mimetypes as _mt

            guessed, _ = _mt.guess_type(filename) or (None, None)
            mimetype = guessed or "application/octet-stream"

        inline_extensions = {".pdf", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
        disposition = "inline" if ext in inline_extensions else "attachment"

        try:
            data = candidate.read_bytes()
            app.logger.info("[UPLOADS] AFTER read_bytes len=%s", len(data))
        except OSError as e:
            app.logger.exception(
                "Erreur lecture fichier upload: %s (errno=%s)",
                candidate,
                getattr(e, "errno", None),
            )
            raise NotFound from e

        # Content-Length explicite : évite ERR_CONNECTION_RESET avec certains clients/proxies
        headers = {
            "Content-Length": str(len(data)),
            "Content-Disposition": f'{disposition}; filename="{candidate.name}"',
            "X-Content-Type-Options": "nosniff",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Cache-Control": "public, max-age=3600",
        }
        origin = request.headers.get("Origin")
        if origin and origin in cors_origins:
            headers["Access-Control-Allow-Origin"] = origin
        return Response(data, mimetype=mimetype, headers=headers)

    # 4) Sécurité (CSP)
    # ✅ FIX: Désactiver HTTPS en mode testing pour éviter
    # les redirections 302 dans les tests E2E
    if config_name in {"testing", "development"}:
        csp = {
            "default-src": "'self'",
            "script-src": "'self'",
            # ✅ 'unsafe-inline' supprimé : styles inline migrés vers CSS
            "style-src": "'self'",
            "img-src": "'self' data: blob:",
            "connect-src": "'self' http://localhost:3000 http://127.00.1:3000 ws: wss:",
        }
        force_https = False
    else:
        # production
        frontend_url = os.getenv("FRONTEND_URL", "https://www.lirie.ch/")
        csp = {
            "default-src": "'self'",
            "script-src": "'self'",
            # ✅ 'unsafe-inline' supprimé : styles inline migrés vers CSS
            "style-src": "'self'",
            "img-src": "'self' data: blob:",
            "connect-src": f"'self' {frontend_url} ws: wss:",
        }
        # En production, on force HTTPS pour la sécurité par défaut
        # (la vérification finale pour l'environnement RL sera faite après)
        # Note: Les healthchecks Docker utilisent HTTP depuis localhost
        # On va créer un endpoint /health exempt de Talisman
        # pour permettre les healthchecks
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
        """Court-circuite Talisman pour /health depuis localhost
        (healthchecks Docker).
        """
        if request.path == "/health":
            # Vérifier si la requête vient de localhost (healthcheck Docker)
            # Dans Docker, remote_addr peut être "127.0.0.1", "::1",
            # ou l'IP du conteneur
            remote = request.remote_addr or ""
            host = request.host or ""
            # Si c'est depuis localhost OU si le schéma est HTTP
            # (pas HTTPS), retourner directement
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
        return None

    # ✅ FIX: Requêtes internes gateway_auth (127.0.0.1/backend) vers auth/login
    # Talisman redirige HTTP→HTTPS, le gateway suit et échoue (SSL sur port 5000).
    # On simule HTTPS pour que Talisman ne redirige pas.
    # X-Internal-Gateway-Auth: header envoyé par gateway_auth._delegate (plus fiable que remote_addr)
    @app.before_request
    def _bypass_talisman_for_gateway_internal_auth():  # pyright: ignore[reportUnusedFunction]
        """Évite la redirection HTTPS pour les appels internes du gateway vers auth."""
        if request.path in ("/api/v1/auth/login", "/api/v1/auth/me"):
            internal_header = request.headers.get("X-Internal-Gateway-Auth") == "1"
            remote = (request.remote_addr or "").strip()
            host = (request.host or "").lower()
            if (
                internal_header
                or remote in ("127.0.0.1", "::1", "localhost")
                or "backend" in host
            ):
                request.environ["wsgi.url_scheme"] = "https"

    # ✅ FIX RC1: Intercepter /api/v1/prometheus/metrics avant Talisman
    @app.before_request
    def _bypass_talisman_for_prometheus_metrics():  # pyright: ignore
        """Court-circuite Talisman pour /api/v1/prometheus/metrics
        (prometheus scraping sans HTTPS).
        """
        if request.path == "/api/v1/prometheus/metrics":
            # Vérifier si la requête vient de localhost ou utilise HTTP
            remote = request.remote_addr or ""
            host = request.host or ""
            # Si c'est depuis localhost OU si le schéma est HTTP
            # (pas HTTPS) OU en mode testing, retourner directement
            if (
                remote in ("127.0.0.1", "::1", "localhost")
                or request.scheme == "http"
                or "localhost" in host
                or current_app.config.get("TESTING", False)
            ):
                # Appeler directement la méthode PrometheusMetrics.get()
                # pour éviter Talisman
                from routes.prometheus_metrics import PrometheusMetrics

                metrics_resource = PrometheusMetrics()
                return metrics_resource.get()
        return None

    @app.before_request
    def _bypass_talisman_for_readiness():  # pyright: ignore[reportUnusedFunction]
        """Healthcheck Docker/Traefik : /api/v1/ready sans redirection HTTPS."""
        if request.path not in ("/api/v1/ready", "/ready"):
            return None
        remote = request.remote_addr or ""
        host = request.host or ""
        if (
            remote in ("127.0.0.1", "::1", "localhost")
            or request.scheme == "http"
            or "localhost" in host
            or current_app.config.get("TESTING", False)
        ):
            from routes.healthcheck import readiness

            return readiness()
        return None

    # ✅ FIX RC1: Désactiver strict_transport_security et force_https en mode testing
    # S'assurer que force_https est bien False en testing (force explicitement)
    if config_name == "testing":
        force_https = False  # Force explicitement pour testing
        strict_transport_security = False
    else:
        strict_transport_security = config_name != "testing"

    # ✅ FIX RC1: Double vérification pour s'assurer que force_https
    # est False en testing (au cas où config_name n'est pas "testing"
    # mais FLASK_CONFIG=testing)
    if (
        app.config.get("TESTING", False)
        or os.getenv("FLASK_CONFIG", default=None) == "testing"
    ):
        force_https = False
        strict_transport_security = False

    # ✅ FIX RL/DEMO: Vérification finale pour désactiver force_https dans
    # les environnements internes non publics (RL et demo).
    # Cette vérification doit être faite APRÈS toutes les autres vérifications
    # pour s'assurer que force_https reste False pour ces environnements.
    app_env = os.getenv("APP_ENV", "").lower()
    rl_enabled = os.getenv("RL_ENABLED", "false").lower() == "true"
    if app_env in {"rl", "demo"} or rl_enabled:
        force_https = False
        strict_transport_security = False
        msg = (
            "[App] Environnement interne détecté (RL/demo): "
            "force_https désactivé pour communication locale"
        )
        app.logger.info(msg)

    # ✅ FIX: Désactiver complètement Talisman pour le mode testing
    # et l'environnement RL
    # pour éviter toute redirection 302
    # Talisman peut encore causer des redirections même avec
    # force_https=False
    # dans certains cas (notamment avec ProxyFix qui détecte
    # X-Forwarded-Proto)
    should_disable_talisman = (
        config_name == "testing"
        or app.config.get("TESTING", False)
        or os.getenv("FLASK_CONFIG", default=None) == "testing"
        or app_env in {"rl", "demo"}
        or rl_enabled
    )

    if should_disable_talisman:
        # Ne pas initialiser Talisman en mode testing ou environnement RL
        talisman = None
        if app_env in {"rl", "demo"} or rl_enabled:
            msg = (
                "[App] Talisman désactivé pour l'environnement interne "
                "(RL/demo, communication HTTP locale)"
            )
            app.logger.info(msg)
        else:
            msg = (
                "[App] Talisman désactivé en mode testing pour éviter "
                "les redirections 302"
            )
            app.logger.info(msg)
    else:
        talisman = Talisman(
            content_security_policy=csp,
            force_https=force_https,
            strict_transport_security=strict_transport_security,
            strict_transport_security_max_age=31536000
            if strict_transport_security
            else 0,  # 1 an
        )
        talisman.init_app(app)

    # Retirer CSP pour les réponses JSON et forcer UTF-8
    # ✅ S2: Ajouter headers de sécurité manquants (Referrer-Policy, Permissions-Policy)
    @app.after_request
    def add_security_headers_and_strip_csp(resp):  # pyright: ignore[reportUnusedFunction]
        # ✅ S2: Ajouter Referrer-Policy: strict-origin-when-cross-origin
        # Limite la fuite d'informations via le header Referer
        if "Referrer-Policy" not in resp.headers:
            resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # ✅ S2: Ajouter Permissions-Policy pour désactiver features
        # non nécessaires
        # Désactive les fonctionnalités du navigateur qui ne sont pas
        # nécessaires pour l'application
        # Cela réduit la surface d'attaque et limite les risques
        # de fuite d'informations
        if "Permissions-Policy" not in resp.headers:
            # Désactiver les fonctionnalités non nécessaires
            # pour une application de transport
            # - geolocation: désactivé (on utilise notre propre système
            # de géolocalisation)
            # - camera: désactivé (pas de caméra nécessaire)
            # - microphone: désactivé (pas de microphone nécessaire)
            # - payment: désactivé (gestion des paiements via API)
            # - usb: désactivé (pas d'accès USB nécessaire)
            # - bluetooth: désactivé (pas d'accès Bluetooth nécessaire)
            # - magnetometer: désactivé (pas de magnétomètre nécessaire)
            # - gyroscope: désactivé (pas de gyroscope nécessaire)
            # - accelerometer: désactivé (pas d'accéléromètre nécessaire)
            permissions_policy = (
                "geolocation=(), "
                "camera=(), "
                "microphone=(), "
                "payment=(), "
                "usb=(), "
                "bluetooth=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "accelerometer=()"
            )
            resp.headers["Permissions-Policy"] = permissions_policy

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

    # ✅ API Versioning: Ajouter header X-API-Version dans toutes les réponses
    @app.after_request
    def add_api_version_header(resp):  # pyright: ignore[reportUnusedFunction]
        """Ajoute le header X-API-Version basé sur le chemin de la requête.

        Extrait la version depuis request.path :
        - /api/v1/* → X-API-Version: v1
        - /api/v2/* → X-API-Version: v2
        - /api/* (legacy) → X-API-Version: legacy
        - Autres chemins → Pas de header (non-API)

        Ce header permet aux clients de détecter automatiquement la version
        de l'API utilisée pour faciliter la migration et le debugging.
        """
        path = request.path or ""

        # Extraire la version depuis le chemin
        # Pattern: /api/v1/, /api/v2/, ou /api/ (legacy)
        match = re.search(r"/api/v(\d+)/", path)
        if match:
            version = f"v{match.group(1)}"
            resp.headers["X-API-Version"] = version
        elif path.startswith("/api/") and not re.search(r"/api/v\d+/", path):
            # Route legacy /api/* (sans /v1/ ou /v2/)
            resp.headers["X-API-Version"] = "legacy"
        # Pour les autres chemins (non-API), ne pas ajouter le header

        return resp

    # ✅ FIX CORS Mobile: Handler after_request pour garantir les headers CORS
    # Sur TOUTES les réponses (OPTIONS, GET, POST, etc.) quand Origin est présent,
    # sinon le navigateur bloque et affiche ERR_NETWORK (POST login 200 / GET profile).
    @app.after_request
    def _ensure_cors_headers_for_options(resp):  # pyright: ignore[reportUnusedFunction]
        """Garantit que les headers CORS sont présents pour OPTIONS et pour les réponses réelles (GET/POST)."""
        origin = request.headers.get("Origin")
        has_acao = "Access-Control-Allow-Origin" in resp.headers
        if request.method == "OPTIONS":
            app.logger.info(
                "[CORS-FIX] after_request pour OPTIONS %s, origin: %s, has_headers: %s",
                request.path,
                origin,
                has_acao,
            )
        # Pour TOUTE requête avec Origin : forcer les en-têtes CORS (origine exacte, jamais * avec credentials).
        if origin and origin in cors_origins:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            )
            # Allow-Headers : une seule source de vérité (echo), jamais "*" (ignoré avec credentials).
            # Si déjà défini (ex. par le handler OPTIONS preflight) → ne pas écraser.
            if "Access-Control-Allow-Headers" not in resp.headers:
                if cors_allow_headers_wildcard:
                    # Dev/test : echo exact de ce que le client a demandé (pas "*").
                    req_headers = request.headers.get(
                        "Access-Control-Request-Headers", ""
                    ).strip()
                    resp.headers["Access-Control-Allow-Headers"] = (
                        req_headers
                        if req_headers
                        else (
                            "Content-Type, Authorization, X-CSRF-Token, "
                            "X-Csrf-Token, X-Requested-With, X-Session-Diag"
                        )
                    )
                else:
                    resp.headers["Access-Control-Allow-Headers"] = (
                        "Content-Type, Authorization, Cache-Control, Pragma, "
                        "X-Device-ID, X-Company-ID, X-Session-ID, "
                        "X-CSRF-Token, X-Csrf-Token, X-Requested-With, "
                        "X-Session-Diag, x-session-diag"
                    )
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Max-Age"] = "3600"
            _vary = resp.headers.get("Vary", "")
            if "Origin" not in _vary:
                resp.headers["Vary"] = (_vary + ", Origin").strip(", ")
            if request.method != "OPTIONS":
                app.logger.info(
                    "[CORS-FIX] Headers CORS ajoutés pour %s %s (évite ERR_NETWORK)",
                    request.method,
                    request.path,
                )
        return resp

    # ✅ FIX CORS Mobile: Handler explicite pour les requêtes OPTIONS preflight
    # Doit être défini AVANT Flask-CORS pour intercepter les requêtes OPTIONS
    @app.before_request
    def _handle_cors_preflight():  # pyright: ignore[reportUnusedFunction]
        """Gère les requêtes OPTIONS preflight pour CORS (avant Flask-CORS)."""
        if request.method == "OPTIONS":
            # Log Flask standard pour vérifier que le handler s'exécute
            app.logger.info(
                "[CORS-PREFLIGHT] Handler appelé pour OPTIONS %s depuis %s",
                request.path,
                request.headers.get("Origin", "no-origin"),
            )
            # Vérifier si l'origine est autorisée
            origin = request.headers.get("Origin")
            app.logger.info(
                "[CORS-PREFLIGHT] Origin: %s, cors_origins: %s, in list: %s",
                origin,
                cors_origins,
                origin in cors_origins if origin else False,
            )
            if origin and origin in cors_origins:
                # ✅ Chrome n'envoie pas la requête réelle (GET/POST) après 204 No Content au preflight OPTIONS.
                # Retourner 200 OK au lieu de 204 pour que Chrome envoie bien le GET/POST qui suit.
                response = make_response("", 200)
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, PATCH, DELETE, OPTIONS"
                )
                if cors_allow_headers_wildcard:
                    # Echo exact de ce que le client demande (dev only). Jamais "*" avec credentials.
                    req_headers = request.headers.get(
                        "Access-Control-Request-Headers", ""
                    ).strip()
                    response.headers["Access-Control-Allow-Headers"] = (
                        req_headers
                        if req_headers
                        else (
                            "Content-Type, Authorization, X-CSRF-Token, "
                            "X-Csrf-Token, X-Requested-With, X-Session-Diag"
                        )
                    )
                else:
                    response.headers["Access-Control-Allow-Headers"] = (
                        "Content-Type, Authorization, Cache-Control, Pragma, "
                        "X-Device-ID, X-Company-ID, X-Session-ID, "
                        "X-CSRF-Token, X-Csrf-Token, X-Requested-With, "
                        "X-Session-Diag, x-session-diag"
                    )
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Max-Age"] = (
                    "600" if cors_allow_headers_wildcard else "3600"
                )
                # Éviter cache preflight (ex. ancien 204) pour que Chrome envoie bien le GET/POST qui suit
                response.headers["Cache-Control"] = "no-store, no-cache"
                response.headers["Pragma"] = "no-cache"
                app.logger.info(
                    "[CORS-PREFLIGHT] Réponse créée avec headers: %s",
                    str(dict(response.headers)),
                )
                return response
            # Si l'origine n'est pas dans cors_origins, laisser Flask-CORS gérer
        return None

    # 5) CORS
    # ✅ Dev local : origines explicites (pas de * + headers custom).
    #    - 8081 (driver web) = Bearer only, withCredentials=false.
    #    - 3000 (dashboard company) = Bearer company_access_token (ou cookies si choisi ; choisir 1 modèle).
    # ✅ Access-Control-Allow-Headers inclut Authorization ; OPTIONS géré partout.
    # ✅ Méthodes : GET, POST, PUT, PATCH, DELETE, OPTIONS.
    # ✅ S1: Validation CORS en production - rejeter "*"
    if config_name == "production":
        # Normaliser cors_origins en liste pour validation
        if isinstance(cors_origins, str):
            origins_list = [cors_origins] if cors_origins else []
        else:
            origins_list = cors_origins if cors_origins else []

        # Vérifier que "*" n'est pas présent
        if "*" in origins_list or cors_origins == "*":
            raise RuntimeError(
                "Configuration CORS invalide pour production. "
                + "Les origines CORS ne peuvent pas être '*' en production. "
                + "Fournissez une liste d'origines autorisées via "
                + "SOCKETIO_CORS_ORIGINS (ex: 'https://app.example.com,"
                + "https://www.example.com')."
            )
        # Vérifier que la liste n'est pas vide
        if not origins_list:
            raise RuntimeError(
                "Configuration CORS invalide pour production. "
                + "SOCKETIO_CORS_ORIGINS doit être défini avec au moins une "
                + "origine autorisée."
            )
        # Logger les origines autorisées pour documentation
        msg = (
            "[Security] CORS configuré pour production avec %d "
            "origine(s) autorisée(s): %s"
        )
        app.logger.info(
            msg,
            len(origins_list),
            ", ".join(origins_list),
        )

    # ✅ Log explicite de la configuration CORS avant initialisation
    app.logger.info(
        "[Flask-CORS] Configuration avec %d origine(s) autorisée(s): %s",
        len(cors_origins),
        ", ".join(cors_origins),
    )

    # Jamais "*" avec credentials : liste explicite partout. En dev le preflight fait echo.
    _cors_allow_headers: list[str] = [
        "Content-Type",
        "Authorization",
        "Cache-Control",
        "Pragma",
        "X-Device-ID",
        "X-Company-ID",
        "X-Session-ID",
        "X-CSRF-Token",
        "X-Csrf-Token",
        "X-Requested-With",
        "X-Session-Diag",
        "x-session-diag",
    ]
    CORS(
        app,
        resources={r"/*": {"origins": "*" if cors_origins == "*" else cors_origins}},
        supports_credentials=True,
        # Expose-Headers : utile seulement si le client lit des headers de réponse (ex. X-CSRF-Token). Pour body JSON seul, pas nécessaire.
        expose_headers=["Content-Type", "Authorization", "X-CSRF-Token"],
        allow_headers=_cors_allow_headers,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    @app.after_request
    def _ensure_cors_headers_on_errors(response):  # pyright: ignore[reportUnusedFunction]
        """Garantit les headers CORS même sur réponses d'erreur Flask/RESTX."""
        origin = request.headers.get("Origin")
        if not origin:
            return response

        allowed_origin: str | None = None
        if cors_origins == "*" or origin in cast("list[str]", cors_origins):
            allowed_origin = origin

        if not allowed_origin:
            return response

        response.headers.setdefault("Access-Control-Allow-Origin", allowed_origin)
        response.headers.setdefault("Access-Control-Allow-Credentials", "true")
        response.headers.setdefault(
            "Access-Control-Allow-Headers", ", ".join(_cors_allow_headers)
        )
        response.headers.setdefault(
            "Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        )
        response.headers.setdefault("Vary", "Origin")
        return response

    # 6) Sentry
    from shared.sentry_init import init_sentry

    init_sentry(environment=config_name, flask=True, celery=True)

    # 7) Logging
    # ✅ S1: Forcer WARNING en production, ignorer APP_LOG_LEVEL si défini à DEBUG/INFO
    if config_name == "production":
        # En production, forcer WARNING minimum pour éviter fuite d'informations
        app_log_level = logging.WARNING
        werkzeug_log_level = logging.ERROR
        msg = (
            "[Security] Niveau de log forcé à WARNING en production "
            "(APP_LOG_LEVEL ignoré pour sécurité)"
        )
        app.logger.warning(msg)
    else:
        # En développement/testing, respecter APP_LOG_LEVEL
        app_log_level = getattr(
            logging, os.getenv("APP_LOG_LEVEL", "INFO").upper(), logging.INFO
        )
        werkzeug_log_level = getattr(
            logging, os.getenv("WERKZEUG_LOG_LEVEL", "ERROR").upper(), logging.ERROR
        )

    app.logger.setLevel(app_log_level)
    # ✅ FIX: Configurer aussi app_logger (utilisé dans fanout.py, core.py, etc.)
    from ext import app_logger

    app_logger.setLevel(app_log_level)
    logging.getLogger("werkzeug").setLevel(werkzeug_log_level)

    # ✅ Ajout filtre PII si activé
    if os.getenv("MASK_PII_LOGS", "true").lower() == "true":
        from shared.logging_utils import PIIFilter

        pii_filter = PIIFilter()
        app.logger.addFilter(pii_filter)
        logging.getLogger("werkzeug").addFilter(pii_filter)
        logging.getLogger().addFilter(pii_filter)

    # ✅ P0-1 : filtrer le bruit bénin kafka-python (« Task is already done! »).
    # Le process Flask/Gunicorn est aussi producteur Kafka (path async tracking).
    from shared.logging_utils import configure_kafka_log_noise

    configure_kafka_log_noise()

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

        # ✅ D2: Importer AuditLog pour qu'Alembic le détecte lors des migrations
        from models import EtaAccuracyLog
        from security.audit_log import AuditLog

        _ = models  # Force l'import pour enregistrer les mappers
        _ = AuditLog  # Force l'import pour qu'Alembic le détecte
        _ = EtaAccuracyLog  # Force l'import pour qu'Alembic le détecte

        configure_mappers()

        # Démo partagée: empêcher les suppressions physiques des données de démonstration.
        from services.demo.soft_delete_guard import register_demo_soft_delete_guard

        register_demo_soft_delete_guard()

        if not skip_routes_init:
            from routes_api import init_namespaces

            init_namespaces(app)

        if not skip_routes_init:
            # ✅ Enhanced healthcheck with DB/Redis checks
            from routes.feature_flags_routes import feature_flags_bp
            from routes.healthcheck import healthcheck_bp
            from routes.ml_monitoring import ml_monitoring_bp
            from routes.proactive_alerts import register_proactive_alerts_routes

            # ✅ Enregistrer healthcheck avec préfixe /api/v1
            app.register_blueprint(healthcheck_bp, url_prefix="/api/v1")
            from routes.internal_tracking import internal_tracking_bp

            app.register_blueprint(internal_tracking_bp)
            from routes.tracking_health import tracking_health_bp

            app.register_blueprint(tracking_health_bp)
            app.register_blueprint(feature_flags_bp)
            app.register_blueprint(ml_monitoring_bp)

            # ✅ Tâche 1: Enregistrer Swagger UI pour visualiser la spec OpenAPI
            from routes.swagger_ui import swagger_ui_bp

            app.register_blueprint(swagger_ui_bp)

            # ✅ Enregistrer les routes d'alertes proactives
            register_proactive_alerts_routes(app)

        if not skip_routes_init:
            # Note: Le handler CORS preflight est défini AVANT Flask-CORS (ligne ~1094)
            # pour intercepter les requêtes OPTIONS en premier

            from middleware.api_contract_headers import register_api_contract_headers

            register_api_contract_headers(app)

            # ✅ 3.2: Shim générique pour /api[/vX]/auth/login si RESTX rate
            # Supporte /api/auth/login (legacy), /api/v1/auth/login
            # et /api/v2/auth/login
            @app.before_request
            def _auth_login_shim_any_version():  # pyright: ignore
                p = request.path or ""
                if request.method in ("POST", "OPTIONS") and re.fullmatch(
                    r"/api(?:/v\d+)?/auth/login", p
                ):
                    # Correspond à /api/auth/login (legacy),
                    # /api/v1/auth/login ou /api/v2/auth/login
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
                if path in {
                    "/",
                    "/health",
                    "/docs",
                    "/favicon.ico",
                    "/robots.txt",
                } or path.startswith(
                    ("/swaggerui/", "/static/", "/uploads/", "/socket.io")
                ):
                    return None

                current_app.logger.debug("Legacy shim: %s %s", request.method, path)

                if any(
                    path == f"/{p}" or path.startswith(f"/{p}/")
                    for p in legacy_prefixes
                ):
                    if request.method == "OPTIONS":
                        return make_response("", 204)
                    new_path = "/api" + path
                    environ = request.environ
                    environ["ORIGINAL_PATH_INFO"] = path
                    environ["PATH_INFO"] = new_path
                    current_app.logger.info(
                        "Legacy shim internal reroute %s %s -> %s",
                        request.method,
                        path,
                        new_path,
                    )
                    return None
                return None

        if not skip_socketio:
            app.logger.info("🔧 [INIT] Enregistrement des handlers Socket.IO chat...")

            # ✅ Enregistrer le vrai handler avec authentification JWT
            try:
                from sockets.chat import init_chat_socket

                app.logger.info("✅ Import init_chat_socket réussi")

                init_chat_socket(socketio)
                app.logger.info("✅ init_chat_socket(socketio) exécuté")
            except Exception as e:
                app.logger.exception(
                    "❌ ERREUR lors de l'enregistrement des handlers Socket.IO: %s", e
                )
                raise

            app.logger.info("✅ Handlers Socket.IO chat enregistrés")

        @app.route("/")
        def index():  # pyright: ignore[reportUnusedFunction]
            return jsonify({"message": "Welcome to the Transport API"}), 200

        # Compat: accès direct à l'autocomplete
        # (certains environnements/proxy peuvent rater la déclaration RESTX)
        @app.route("/api/geocode/autocomplete", methods=["GET", "OPTIONS"])
        def _compat_geocode_autocomplete():  # pyright: ignore
            if request.method == "OPTIONS":
                return make_response("", 204)
            try:
                from routes.geocode import GeocodeAutocomplete

                res = GeocodeAutocomplete()
                return res.get()
            except Exception as err:
                # Laisse la 404 standard si la ressource n'est pas dispo
                raise NotFound from err

        # Compat: accès direct au login
        # (certains environnements/proxy peuvent rater la déclaration RESTX)
        from routes.auth import Login, LoginTest  # Import au niveau module

        @app.route("/api/auth/login", methods=["POST", "OPTIONS"])
        def _compat_auth_login():  # pyright: ignore
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
        def _compat_auth_login_v(version: int):  # pyright: ignore  # noqa: ARG001
            if request.method == "OPTIONS":
                return make_response("", 204)
            return Login().post()

        # Compat: accès direct au login-test (pour tests de charge)
        @app.route("/api/auth/login-test", methods=["POST", "OPTIONS"])
        def _compat_auth_login_test():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            return LoginTest().post()

        @app.route("/api/v<int:version>/auth/login-test", methods=["POST", "OPTIONS"])
        def _compat_auth_login_test_v(version: int):  # pyright: ignore  # noqa: ARG001
            if request.method == "OPTIONS":
                return make_response("", 204)
            return LoginTest().post()

        # --- Compat: endpoints companies les plus utilisés
        # (évite 404 si RESTX rate) ---
        from routes.companies import (
            CompanyDriversList,
            CompanyDriversLocations,
            CompanyMe,
        )

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

        @app.route(
            "/api/v<int:version>/companies/me", methods=["GET", "PUT", "OPTIONS"]
        )
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

        @app.route("/api/companies/me/drivers/locations", methods=["GET", "OPTIONS"])
        def _compat_companies_me_drivers_locations():  # pyright: ignore[reportUnusedFunction]
            if request.method == "OPTIONS":
                return make_response("", 204)
            return CompanyDriversLocations().get()

        @app.route(
            "/api/v<int:version>/companies/me/drivers", methods=["GET", "OPTIONS"]
        )
        def _compat_companies_me_drivers_v(  # pyright: ignore[reportUnusedFunction]
            version: int,  # noqa: ARG001
        ):
            if request.method == "OPTIONS":
                return make_response("", 204)
            return CompanyDriversList().get()

        @app.route(
            "/api/v<int:version>/companies/me/drivers/locations",
            methods=["GET", "OPTIONS"],
        )
        def _compat_companies_me_drivers_locations_v(  # pyright: ignore[reportUnusedFunction]
            version: int,  # noqa: ARG001
        ):
            if request.method == "OPTIONS":
                return make_response("", 204)
            return CompanyDriversLocations().get()

        # Note: L'endpoint /health est défini plus bas,
        # après l'initialisation de Talisman
        # Il sera exempt de la redirection HTTPS via un décorateur
        # ou une configuration spéciale

        noisy_paths = {
            "/companies/me/dispatch/status",
            "/company_dispatch/status",
        }

        @app.after_request
        def log_request_info(response):  # pyright: ignore[reportUnusedFunction]
            if request.path in noisy_paths and request.method in ("GET", "OPTIONS"):
                return response
            app.logger.debug(
                "%s %s -> %s", request.method, request.path, response.status_code
            )
            return response

        # JWT : handlers d'erreurs
        @jwt.expired_token_loader
        def expired_token_callback(_jwt_header, _jwt_payload):  # pyright: ignore
            return jsonify(
                {"error": "token_expired", "message": "Signature has expired"}
            ), 401

        @jwt.invalid_token_loader
        def invalid_token_callback(error):  # pyright: ignore[reportUnusedFunction]
            return jsonify({"error": "invalid_token", "message": str(error)}), 422

        @jwt.unauthorized_loader
        def missing_token_callback(error):  # pyright: ignore[reportUnusedFunction]
            return jsonify({"error": "missing_token", "message": str(error)}), 401

        @app.errorhandler(BadRequest)
        def handle_bad_request(e: BadRequest):  # pyright: ignore
            """Handler BadRequest aligné sur le format standard v2.

            Format: {"error": "error_code", "message": "Message"}
            """
            original_message = getattr(e, "description", str(e))
            app.logger.warning(
                "[BadRequest] path=%s method=%s Content-Type=%s Content-Length=%s desc=%s",
                getattr(request, "path", None),
                getattr(request, "method", None),
                getattr(request, "content_type", None),
                getattr(request, "content_length", None),
                original_message,
            )

            # Flask-RESTX abort(400) : message dans e.data, description Werkzeug générique
            payload_data = getattr(e, "data", None)
            if isinstance(payload_data, dict):
                restx_message = payload_data.get("message")
                restx_errors = payload_data.get("errors")
                if restx_message or restx_errors:
                    first_field_error = None
                    if isinstance(restx_errors, dict) and restx_errors:
                        first_val = next(iter(restx_errors.values()))
                        if isinstance(first_val, list) and first_val:
                            first_field_error = str(first_val[0])
                        elif first_val is not None:
                            first_field_error = str(first_val)
                    error_message = (
                        first_field_error or restx_message or "Données invalides"
                    )
                    response_body: dict[str, object] = {
                        "error": "validation_error",
                        "message": error_message,
                    }
                    if restx_errors:
                        response_body["details"] = {"errors": restx_errors}
                    return jsonify(response_body), 400

            # Déterminer le code d'erreur et le message
            werkzeug_default = "The browser (or proxy) sent a request that this server could not understand."

            if original_message == werkzeug_default:
                # Erreur générique Werkzeug - clarifier selon le contexte
                request_mimetype = (getattr(request, "mimetype", None) or "").lower()
                if request_mimetype != "application/json":
                    error_code = "invalid_content_type"
                    error_message = f"Content-Type invalide: '{request.content_type}'. Utilisez 'application/json'."
                else:
                    import json as _json

                    from middleware.silent_json_request import redact_json_body_preview

                    raw_body = request.get_data(cache=True, as_text=True)
                    body_parses = False
                    if raw_body and raw_body.strip():
                        try:
                            _json.loads(raw_body)
                            body_parses = True
                        except _json.JSONDecodeError:
                            body_parses = False

                    if body_parses:
                        error_code = "bad_request"
                        error_message = "Requête invalide. Vérifiez le format et les types des champs."
                    else:
                        app.logger.warning(
                            "[BadRequest] JSON non parseable path=%s body_len=%s preview=%s",
                            getattr(request, "path", None),
                            len(raw_body) if raw_body else 0,
                            redact_json_body_preview(raw_body),
                        )
                        error_code = "invalid_json"
                        error_message = (
                            "Corps de requête JSON manquant ou invalide. "
                            "Vérifiez le format du body."
                        )
            else:
                # Message spécifique - déterminer le code approprié
                msg_lower = original_message.lower()
                if (
                    "password" in msg_lower
                    or "email" in msg_lower
                    or "required" in msg_lower
                    or "missing" in msg_lower
                ):
                    error_code = "validation_error"
                else:
                    error_code = "bad_request"
                error_message = original_message

            return jsonify(
                {
                    "error": error_code,
                    "message": error_message,
                }
            ), 400

        @app.errorhandler(HTTPException)
        def handle_http_exception(e: HTTPException):  # pyright: ignore
            """Handler HTTPException aligné sur le format standard v2.

            Format: {"error": "error_code", "message": "Message", "details": {}}
            """
            if isinstance(e, NotFound):
                app.logger.warning("404 on path: %s", request.path)

            status_code: int = int(e.code or 500)

            # Convertir le nom HTTP en code machine (snake_case)
            # Ex: "Bad Request" -> "bad_request", "Not Found" -> "not_found"
            error_code = (
                e.name.lower().replace(" ", "_") if e.name else f"http_{status_code}"
            )

            return jsonify(
                {
                    "error": error_code,
                    "message": e.description or f"HTTP {status_code} error",
                }
            ), status_code

        @app.errorhandler(Exception)
        def handle_exception(e: Exception):  # pyright: ignore[reportUnusedFunction]
            # Intercepter les erreurs JWT expirées qui peuvent échapper aux handlers JWT
            from jwt.exceptions import (
                ExpiredSignatureError,
                InvalidTokenError,
            )

            if isinstance(e, ExpiredSignatureError):
                app.logger.warning("Token JWT expiré intercepté: %s", str(e))
                return jsonify(
                    {
                        "error": "token_expired",
                        "message": "Signature has expired",
                    }
                ), 401
            if isinstance(e, InvalidTokenError):
                app.logger.warning("Token JWT invalide intercepté: %s", str(e))
                return jsonify({"error": "invalid_token", "message": str(e)}), 422

            # ✅ S1: Réduire les détails dans les stack traces en production
            # En production, ne jamais exposer les détails de l'exception
            is_production = config_name == "production"
            is_debug = app.config.get("DEBUG", False)

            # Logger l'exception complète (pour Sentry/logs serveur)
            app.logger.exception("Unhandled server error")

            # Message pour le client : générique en production,
            # détaillé seulement en dev
            if is_production or not is_debug:
                # Production ou non-DEBUG : message générique
                msg = "Une erreur interne est survenue."
            else:
                # Développement avec DEBUG : message détaillé (pour debugging)
                msg = str(e)

            return jsonify({"error": "server_error", "message": msg}), 500

    # Note: handler disconnect géré dans sockets/chat.py (pas de doublon)

    from cli.platform_billing_cli import register_platform_billing_cli

    register_platform_billing_cli(app)

    return app
